from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests
import torch
import time
import psutil
import os

# Load BLIP processor and model
processor = BlipProcessor.from_pretrained("blip-image-captioning-base", use_fast=True)
model = BlipForConditionalGeneration.from_pretrained("blip-image-captioning-base", torch_dtype=torch.float16)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total Parameters: {total_params/1000000:,}")
print(f"Trainable Parameters: {trainable_params:,}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load an image
url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

inputs = processor(images=image, return_tensors="pt").to(device)

cpu_before = psutil.cpu_percent(interval=None)

start = time.time()
with torch.no_grad():
    out = model.generate(**inputs)
inference_time = time.time() - start

cpu_after = psutil.cpu_percent(interval=None)

if torch.cuda.is_available():
    gpu_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # in MB
else:
    gpu_mem = None

'''# Generate caption
inputs = processor(images=image, return_tensors="pt")
out = model.generate(**inputs)
caption = processor.decode(out[0], skip_special_tokens=True)'''

caption = processor.decode(out[0], skip_special_tokens=True)
print("Generated Caption:", caption)
print(f"Inference Time: {inference_time:.4f} seconds")
print(f"CPU Usage Change: {cpu_after - cpu_before:.2f}%")

if gpu_mem is not None:
    print(f"Max GPU Memory Used: {gpu_mem:.2f} MB")
else:
    print("GPU not available.")