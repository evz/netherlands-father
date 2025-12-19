import argparse
import os
import time
import requests
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class OpenAIVideoClient:
    """Client for OpenAI's Sora API."""

    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPEN_AI_API_KEY"))

    def generate(self, prompt: str, seconds: int = 4, resolution: str = "1280x720", fps: int = 24) -> bytes:
        print("Submitting video generation request to OpenAI...")

        video = self.client.videos.create(
            model="sora-2",
            prompt=prompt,
            seconds=str(seconds),
            size=resolution,
        )

        video_id = video.id
        print(f"Video job created: {video_id}")
        print("Waiting for generation to complete...")

        while True:
            status = self.client.videos.retrieve(video_id)

            if status.status == "completed":
                print("Video generated successfully!")
                print("Fetching video content...")
                return self.client.videos.download_content(video_id).read()
            elif status.status == "failed":
                raise Exception(f"Video generation failed: {status.error}")

            print(f"Status: {status.status}...")
            time.sleep(5)

    def fetch(self, video_id: str) -> bytes:
        print(f"Fetching video {video_id} from OpenAI...")
        return self.client.videos.download_content(video_id).read()


class LocalVideoClient:
    """Client for local Open-Sora server."""

    def __init__(self, server_url: str):
        self.server_url = server_url.rstrip("/")

    def generate(self, prompt: str, seconds: int = 4, resolution: str = "1280x720", fps: int = 24) -> bytes:
        print(f"Submitting video generation request to {self.server_url}...")

        response = requests.post(
            f"{self.server_url}/videos",
            json={
                "prompt": prompt,
                "seconds": seconds,
                "size": resolution,
                "fps": fps,
            },
        )
        response.raise_for_status()
        job = response.json()

        video_id = job["id"]
        print(f"Video job created: {video_id}")
        print("Waiting for generation to complete...")

        while True:
            status_response = requests.get(f"{self.server_url}/videos/{video_id}")
            status_response.raise_for_status()
            status = status_response.json()

            if status["status"] == "completed":
                print("Video generated successfully!")
                print("Fetching video content...")
                return self._download(video_id)
            elif status["status"] == "failed":
                raise Exception(f"Video generation failed: {status.get('error')}")

            progress = status.get("progress", 0)
            print(f"Status: {status['status']} ({progress}%)...")
            time.sleep(5)

    def fetch(self, video_id: str) -> bytes:
        print(f"Fetching video {video_id} from {self.server_url}...")
        return self._download(video_id)

    def _download(self, video_id: str) -> bytes:
        response = requests.get(f"{self.server_url}/videos/{video_id}/content")
        response.raise_for_status()
        return response.content


def save_video(content: bytes, output_path: str | None = None) -> str:
    """Save video content to a file."""
    if output_path is None:
        timestamp = int(time.time())
        output_path = f"generated_video_{timestamp}.mp4"

    print(f"Saving video to {output_path}...")

    with open(output_path, "wb") as f:
        f.write(content)

    print(f"Saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate videos using Sora or Open-Sora")
    parser.add_argument("prompt", nargs="?", help="Video prompt")
    parser.add_argument("-f", "--file", help="Read prompt from file")
    parser.add_argument("-d", "--duration", type=int, default=4,
                        help="Video duration in seconds (default: 4)")
    parser.add_argument("-r", "--resolution", default="1280x720",
                        help="Output resolution (default: 1280x720)")
    parser.add_argument("--fps", type=int, default=24,
                        help="Frames per second (default: 24, local only)")
    parser.add_argument("-o", "--output", help="Output filename")
    parser.add_argument("--fetch", metavar="VIDEO_ID", help="Fetch existing video by ID")

    # Backend selection
    parser.add_argument("--local", action="store_true",
                        help="Use local Open-Sora server instead of OpenAI")
    parser.add_argument("--server", default=os.getenv("OPENSORA_SERVER", "http://localhost:8000"),
                        help="Local server URL (default: http://localhost:8000)")

    args = parser.parse_args()

    # Select client
    if args.local:
        client = LocalVideoClient(args.server)
    else:
        client = OpenAIVideoClient()

    output_file = args.output
    if output_file and not output_file.endswith(".mp4"):
        output_file = f"{output_file}.mp4"

    if args.fetch:
        content = client.fetch(args.fetch)
    else:
        if args.file:
            with open(args.file) as f:
                prompt = f.read().strip()
        else:
            prompt = args.prompt or input("Enter your video prompt: ")
        content = client.generate(
            prompt,
            seconds=args.duration,
            resolution=args.resolution,
            fps=args.fps,
        )

    save_video(content, output_file)
