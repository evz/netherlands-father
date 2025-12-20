import os
import sqlite3
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel

# SANA directory
SANA_DIR = Path(os.getenv("SANA_DIR", "/opt/Sana"))

app = FastAPI(title="SANA-Video Generation API")

# Configuration
DB_PATH = Path(__file__).parent / "jobs.db"
OUTPUT_DIR = Path(__file__).parent / "outputs"
INFERENCE_SCRIPT = Path(__file__).parent / "sana_inference.py"
# SANA model ID from HuggingFace
SANA_MODEL_ID = os.getenv("SANA_MODEL_ID", "Efficient-Large-Model/SANA-Video_2B_480p_diffusers")
# Python executable (use the one from venv if available)
PYTHON_BIN = os.getenv("PYTHON_BIN", sys.executable)

OUTPUT_DIR.mkdir(exist_ok=True)


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
    """Process a video generation job using SANA-Video via subprocess."""
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
        # Parse resolution
        width, height = map(int, size.split("x"))

        # Calculate motion score based on video duration (longer videos = higher motion)
        # For 4s video use motion score 30, scale proportionally
        motion_score = min(100, max(10, int(30 * (seconds / 4.0))))

        # Build command to run inference script as subprocess
        cmd = [
            PYTHON_BIN,
            str(INFERENCE_SCRIPT),
            "--prompt", prompt,
            "--output", str(output_path),
            "--model-id", SANA_MODEL_ID,
            "--height", str(height),
            "--width", str(width),
            "--frames", str(num_frames),
            "--fps", str(fps),
            "--motion-score", str(motion_score),
        ]

        with get_db() as conn:
            conn.execute(
                "UPDATE jobs SET progress = ? WHERE id = ?",
                (30, job_id),
            )

        # Run inference in subprocess
        print(f"[INFO] Running inference subprocess for job {job_id}")
        print(f"[INFO] Command: {' '.join(cmd)}")

        env = os.environ.copy()
        env["SANA_DIR"] = str(SANA_DIR)

        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Stream output and update progress
        stdout_lines = []
        stderr_lines = []

        while True:
            # Read stdout
            stdout_line = process.stdout.readline()
            if stdout_line:
                print(stdout_line.rstrip())
                stdout_lines.append(stdout_line)

                # Update progress based on log messages
                if "Models loaded successfully" in stdout_line or "Loading SANA-Video pipeline" in stdout_line:
                    with get_db() as conn:
                        conn.execute("UPDATE jobs SET progress = ? WHERE id = ?", (50, job_id))
                elif "Starting inference" in stdout_line:
                    with get_db() as conn:
                        conn.execute("UPDATE jobs SET progress = ? WHERE id = ?", (60, job_id))
                elif "Inference completed" in stdout_line:
                    with get_db() as conn:
                        conn.execute("UPDATE jobs SET progress = ? WHERE id = ?", (80, job_id))
                elif "Saving video" in stdout_line:
                    with get_db() as conn:
                        conn.execute("UPDATE jobs SET progress = ? WHERE id = ?", (90, job_id))

            # Check if process finished
            if process.poll() is not None:
                # Read remaining output
                remaining_stdout = process.stdout.read()
                if remaining_stdout:
                    print(remaining_stdout.rstrip())
                    stdout_lines.append(remaining_stdout)
                break

        # Read any stderr
        stderr = process.stderr.read()
        if stderr:
            print(f"[STDERR]\n{stderr}", file=sys.stderr)
            stderr_lines.append(stderr)

        returncode = process.returncode

        if returncode != 0:
            error_msg = f"Inference failed with exit code {returncode}"
            if stderr_lines:
                error_msg += f"\n{stderr}"
            raise RuntimeError(error_msg)

        # Check if the video file was created
        if not Path(output_path).exists():
            raise RuntimeError(f"Video file not found at {output_path}")

        with get_db() as conn:
            conn.execute(
                "UPDATE jobs SET status = ?, progress = ?, completed_at = ?, file_path = ? WHERE id = ?",
                ("completed", 100, int(time.time()), str(output_path), job_id),
            )

        print(f"[INFO] Job {job_id} completed successfully")

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
    print("[INFO] SANA-Video server started. Video generation will run in isolated subprocesses.")
    print(f"[INFO] Using model: {SANA_MODEL_ID}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
