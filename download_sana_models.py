#!/usr/bin/env python3
"""
Pre-download SANA-Video models from HuggingFace.
This will cache the models so the first inference is faster.
"""
import torch
from diffusers import SanaVideoPipeline

print("Downloading SANA-Video models...")
print("This may take a few minutes depending on your connection.")
print()

model_id = "Efficient-Large-Model/SANA-Video_2B_480p_diffusers"
print(f"Model: {model_id}")

# Download the pipeline (this caches all components)
pipe = SanaVideoPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
)

print()
print("✓ Models downloaded successfully!")
print(f"Cached in: ~/.cache/huggingface/hub/")
print()
print("You can now run your server without waiting for downloads.")
