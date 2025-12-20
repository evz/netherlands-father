#!/usr/bin/env python3
"""
Standalone SANA-Video inference script.
Run video generation in isolation, then exit to free all GPU memory.
"""
import argparse
import os
import sys
import time
from pathlib import Path

# Set environment variables BEFORE importing torch
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
num_threads = str(os.cpu_count() or 1)
os.environ.setdefault('OMP_NUM_THREADS', num_threads)
os.environ.setdefault('MKL_NUM_THREADS', num_threads)
os.environ.setdefault('NUMEXPR_NUM_THREADS', num_threads)

import torch
from diffusers import SanaVideoPipeline
from diffusers.utils import export_to_video


def main():
    parser = argparse.ArgumentParser(description="Generate video using SANA-Video")
    parser.add_argument("--prompt", required=True, help="Text prompt for video generation")
    parser.add_argument("--output", required=True, help="Output file path (with .mp4 extension)")
    parser.add_argument("--model-id", default="Efficient-Large-Model/SANA-Video_2B_480p_diffusers",
                       help="Model ID from HuggingFace")
    parser.add_argument("--height", type=int, default=480, help="Video height")
    parser.add_argument("--width", type=int, default=832, help="Video width")
    parser.add_argument("--frames", type=int, default=81, help="Number of frames")
    parser.add_argument("--fps", type=int, default=16, help="Frames per second")
    parser.add_argument("--guidance-scale", type=float, default=6.0, help="Guidance scale")
    parser.add_argument("--num-inference-steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--motion-score", type=int, default=30, help="Motion score (0-100)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--negative-prompt", type=str,
                       default="A chaotic sequence with misshapen, deformed limbs in heavy motion blur, "
                               "sudden disappearance, jump cuts, jerky movements, rapid shot changes, "
                               "frames out of sync, inconsistent character shapes, temporal artifacts, "
                               "jitter, and ghosting effects, creating a disorienting visual experience.",
                       help="Negative prompt for quality control")

    args = parser.parse_args()

    print(f"[INFO] Starting SANA-Video generation...")
    print(f"[INFO] Prompt: {args.prompt}")
    print(f"[INFO] Model: {args.model_id}")
    print(f"[INFO] Resolution: {args.height}x{args.width}")
    print(f"[INFO] Frames: {args.frames} @ {args.fps}fps")
    print(f"[INFO] Motion score: {args.motion_score}")

    # Setup device and dtype
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("[WARNING] CUDA not available, using CPU (this will be very slow)")

    print(f"[INFO] Using device: {device}")

    # Load pipeline
    print("[INFO] Loading SANA-Video pipeline...")
    try:
        pipe = SanaVideoPipeline.from_pretrained(
            args.model_id,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )

        # Set dtype for components
        if torch.cuda.is_available():
            pipe.vae.to(torch.float32)
            pipe.text_encoder.to(torch.bfloat16)

        pipe.to(device)
        print("[INFO] Models loaded successfully")

    except Exception as e:
        print(f"[ERROR] Failed to load pipeline: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return 1

    # Add motion score to prompt
    motion_prompt = f" motion score: {args.motion_score}."
    full_prompt = args.prompt + motion_prompt

    # Generate seed
    seed = args.seed if args.seed is not None else int(time.time()) % 2**32
    print(f"[INFO] Using seed: {seed}")
    generator = torch.Generator(device=device).manual_seed(seed)

    # Run inference
    print("[INFO] Starting inference...")
    start_time = time.time()

    try:
        with torch.no_grad():
            output = pipe(
                prompt=full_prompt,
                negative_prompt=args.negative_prompt,
                height=args.height,
                width=args.width,
                frames=args.frames,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                generator=generator,
            )

        elapsed = time.time() - start_time
        print(f"[INFO] Inference completed in {elapsed:.1f}s")

        # Save video
        print(f"[INFO] Saving video to {args.output}")
        video_frames = output.frames[0]
        export_to_video(video_frames, args.output, fps=args.fps)

        print(f"[SUCCESS] Video saved to: {args.output}")

        # Clean up
        del output
        del video_frames
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return 0

    except Exception as e:
        import traceback
        print(f"[ERROR] Generation failed: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)

        # Clean up even on error
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return 1


if __name__ == "__main__":
    sys.exit(main())
