from transformers import AutoProcessor, GitForCausalLM
from PIL import Image
import requests, torch

processor = AutoProcessor.from_pretrained("git-base", use_fast=True)
model = GitForCausalLM.from_pretrained("git-base").to("cuda" if torch.cuda.is_available() else "cpu")

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total Parameters: {total_params/1000000:,}")
print(f"Trainable Parameters: {trainable_params:,}")

image = Image.open(requests.get("https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg", stream=True).raw)
inputs = processor(images=image, return_tensors="pt").to(model.device)
generated_ids = model.generate(**inputs, max_new_tokens=50)
print(processor.batch_decode(generated_ids, skip_special_tokens=True)[0])
