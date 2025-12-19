# Video Generation CLI

Generate videos using OpenAI's Sora API or a self-hosted Open-Sora server.

## Setup

```bash
pip install openai python-dotenv requests
```

Create a `.env` file:
```
OPEN_AI_API_KEY=sk-...
OPENSORA_SERVER=http://192.168.1.100:8000  # optional
```

## Usage

### OpenAI Sora

```bash
# Basic usage
python generate_video.py "A cat riding a bicycle" -o cat.mp4

# From prompt file
python generate_video.py -f prompt.txt -d 8 -o output.mp4

# Fetch existing video by ID
python generate_video.py --fetch video_abc123 -o download.mp4
```

### Local Open-Sora Server

```bash
# Use local server
python generate_video.py --local "A cat riding a bicycle" -d 10 -o cat.mp4

# Custom server URL
python generate_video.py --local --server http://192.168.1.100:8000 -f prompt.txt -o output.mp4
```

### Options

| Flag | Description |
|------|-------------|
| `-f, --file` | Read prompt from file |
| `-d, --duration` | Video duration in seconds (default: 4) |
| `-r, --resolution` | Output resolution (default: 1280x720) |
| `--fps` | Frames per second (default: 24, local only) |
| `-o, --output` | Output filename |
| `--fetch` | Fetch existing video by ID |
| `--local` | Use local Open-Sora server |
| `--server` | Local server URL |

## Self-Hosted Server Setup

See [server/README.md](server/) for setting up Open-Sora on a GPU machine.

### Quick Start

1. Copy `server/` to your GPU machine
2. Run `./setup.sh`
3. Start: `sudo systemctl start opensora-server`

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/videos` | Create video generation job |
| GET | `/videos/{id}` | Get job status |
| GET | `/videos/{id}/content` | Download completed video |
| GET | `/videos` | List all jobs |
| DELETE | `/videos/{id}` | Delete video and job |
