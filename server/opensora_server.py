import os
import sqlite3
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Add Open-Sora to path before importing
OPENSORA_DIR = Path(os.getenv("OPENSORA_DIR", "/opt/Open-Sora"))
sys.path.insert(0, str(OPENSORA_DIR))

from opensora.datasets.dataloader import prepare_dataloader
from opensora.datasets.utils import save_video
from opensora.models.diffusion import prepare_models, prepare_api
from opensora.models.diffusion.sampling import SamplingOption, sanitize_sampling_option
from opensora.utils.config import parse_configs

app = FastAPI(title="Open-Sora Video Generation API")

# Configuration
DB_PATH = Path(__file__).parent / "jobs.db"
OUTPUT_DIR = Path(__file__).parent / "outputs"
OPENSORA_CONFIG = os.getenv("OPENSORA_CONFIG", "configs/diffusion/inference/t2i2v_256px.py")

OUTPUT_DIR.mkdir(exist_ok=True)

# Global model state (loaded once at startup)
model_state = {
    "api_fn": None,
    "cfg": None,
    "device": None,
    "dtype": None,
    "loaded": False,
}
model_lock = threading.Lock()


def load_models():
    """Load Open-Sora models (called once at startup)."""
    global model_state

    if model_state["loaded"]:
        return

    print("Loading Open-Sora models...")

    # Parse config
    config_path = OPENSORA_DIR / OPENSORA_CONFIG
    cfg = parse_configs([str(config_path)])

    # Setup device and dtype
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # Load models
    model, model_ae, model_t5, model_clip, optional_models = prepare_models(
        cfg, device, dtype, offload_model=False
    )

    # Create API function
    api_fn = prepare_api(model, model_ae, model_t5, model_clip, optional_models)

    model_state["api_fn"] = api_fn
    model_state["cfg"] = cfg
    model_state["device"] = device
    model_state["dtype"] = dtype
    model_state["loaded"] = True

    print("Models loaded successfully!")


# Database setup
def init_db():
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                prompt TEXT NOT NULL,
                seconds INTEGER NOT NULL,
                size TEXT NOT NULL,
                fps INTEGER NOT NULL,
                status TEXT NOT NULL,
                progress INTEGER DEFAULT 0,
                created_at INTEGER NOT NULL,
                completed_at INTEGER,
                error TEXT,
                file_path TEXT
            )
        """)


@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


# Request/Response models
class VideoCreateRequest(BaseModel):
    prompt: str
    seconds: int = 4
    size: str = "1280x720"
    fps: int = 24


class VideoResponse(BaseModel):
    id: str
    prompt: str
    seconds: int
    size: str
    fps: int
    status: str
    progress: int
    created_at: int
    completed_at: Optional[int] = None
    error: Optional[str] = None


class VideoListResponse(BaseModel):
    data: list[VideoResponse]


def row_to_response(row: sqlite3.Row) -> VideoResponse:
    return VideoResponse(
        id=row["id"],
        prompt=row["prompt"],
        seconds=row["seconds"],
        size=row["size"],
        fps=row["fps"],
        status=row["status"],
        progress=row["progress"],
        created_at=row["created_at"],
        completed_at=row["completed_at"],
        error=row["error"],
    )


# Background worker
def process_video_job(job_id: str):
    """Process a video generation job using Open-Sora."""
    with get_db() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if not row:
            return

        prompt = row["prompt"]
        seconds = row["seconds"]
        size = row["size"]
        fps = row["fps"]

    # Update status to processing
    with get_db() as conn:
        conn.execute(
            "UPDATE jobs SET status = ?, progress = ? WHERE id = ?",
            ("in_progress", 10, job_id),
        )

    output_path = OUTPUT_DIR / f"{job_id}.mp4"
    num_frames = seconds * fps

    try:
        with model_lock:
            api_fn = model_state["api_fn"]
            cfg = model_state["cfg"]

            if api_fn is None:
                raise RuntimeError("Models not loaded")

            with get_db() as conn:
                conn.execute(
                    "UPDATE jobs SET progress = ? WHERE id = ?",
                    (30, job_id),
                )

            # Parse resolution
            width, height = map(int, size.split("x"))

            # Setup sampling options
            sampling_option = SamplingOption(
                num_frames=num_frames,
                fps=fps,
                height=height,
                width=width,
            )
            sampling_option = sanitize_sampling_option(sampling_option, cfg)

            # Generate video
            seed = int(time.time()) % 2**32
            batch = {"text": [prompt]}

            with get_db() as conn:
                conn.execute(
                    "UPDATE jobs SET progress = ? WHERE id = ?",
                    (50, job_id),
                )

            # Run inference
            video_tensor = api_fn(
                sampling_option,
                cond_type="t2v",
                seed=seed,
                **batch,
            )

            with get_db() as conn:
                conn.execute(
                    "UPDATE jobs SET progress = ? WHERE id = ?",
                    (80, job_id),
                )

            # Save video
            save_video(video_tensor[0], str(output_path), fps=fps)

        with get_db() as conn:
            conn.execute(
                "UPDATE jobs SET status = ?, progress = ?, completed_at = ?, file_path = ? WHERE id = ?",
                ("completed", 100, int(time.time()), str(output_path), job_id),
            )

    except Exception as e:
        import traceback
        traceback.print_exc()
        with get_db() as conn:
            conn.execute(
                "UPDATE jobs SET status = ?, error = ? WHERE id = ?",
                ("failed", str(e), job_id),
            )


# API Endpoints
@app.post("/videos", response_model=VideoResponse)
def create_video(request: VideoCreateRequest, background_tasks: BackgroundTasks):
    """Create a new video generation job."""
    job_id = f"video_{uuid.uuid4().hex}"
    created_at = int(time.time())

    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO jobs (id, prompt, seconds, size, fps, status, progress, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (job_id, request.prompt, request.seconds, request.size, request.fps, "pending", 0, created_at),
        )

    # Start processing in background
    background_tasks.add_task(process_video_job, job_id)

    return VideoResponse(
        id=job_id,
        prompt=request.prompt,
        seconds=request.seconds,
        size=request.size,
        fps=request.fps,
        status="pending",
        progress=0,
        created_at=created_at,
    )


@app.get("/videos/{video_id}", response_model=VideoResponse)
def get_video(video_id: str):
    """Get the status of a video generation job."""
    with get_db() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (video_id,)).fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Video not found")

    return row_to_response(row)


@app.get("/videos/{video_id}/content")
def download_video(video_id: str):
    """Download a completed video."""
    with get_db() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (video_id,)).fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Video not found")

    if row["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Video not ready. Status: {row['status']}")

    file_path = Path(row["file_path"])
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found")

    return FileResponse(
        file_path,
        media_type="video/mp4",
        filename=f"{video_id}.mp4",
    )


@app.get("/videos", response_model=VideoListResponse)
def list_videos(limit: int = 20):
    """List recent video generation jobs."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()

    return VideoListResponse(data=[row_to_response(row) for row in rows])


@app.delete("/videos/{video_id}")
def delete_video(video_id: str):
    """Delete a video and its job record."""
    with get_db() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (video_id,)).fetchone()

        if not row:
            raise HTTPException(status_code=404, detail="Video not found")

        # Delete file if exists
        if row["file_path"]:
            file_path = Path(row["file_path"])
            if file_path.exists():
                file_path.unlink()

        conn.execute("DELETE FROM jobs WHERE id = ?", (video_id,))

    return {"status": "deleted", "id": video_id}


@app.on_event("startup")
def startup():
    init_db()
    load_models()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
