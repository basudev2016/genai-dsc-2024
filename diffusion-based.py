from diffusers import StableDiffusionPipeline
import torch
from PIL import Image

# Load the pre-trained Stable Diffusion model with float32 for CPU compatibility
# pipeline = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float32)
pipeline = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float32, low_cpu_mem_usage=True)

pipeline = pipeline.to("cpu")  # Ensures the model runs on CPU

# Define your text prompt
prompt = "A fantasy landscape with mountains and a river"
# prompt = "A dog jumpong over a fence"

# Generate the image
with torch.no_grad():
    image = pipeline(prompt).images[0]

# Display the image
image.show()

# Optionally, save the image to a file
image.save("generated_image-diffuon-based.png")
