from transformers import BlipProcessor, BlipForConditionalGeneration

# Local path to the folder
model_path = "blip-image-captioning-base"

# Load processor and model
processor = BlipProcessor.from_pretrained(model_path, use_fast=True)
model = BlipForConditionalGeneration.from_pretrained(model_path)