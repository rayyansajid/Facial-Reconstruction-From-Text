from diffusers import StableDiffusionPipeline
from PIL import Image

# Load the pipeline
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")

# Your text prompt from the CSV
prompt ="This person has a medium skin tone and an asymmetric face shape. Their hair is gray, and they have a pointed nose. Their eyes are round in shape and wide with a surprised or alert expression. The lips are heart-shaped and are slightly parted, giving the impression that they are about to speak. The overall appearance is that of a 36-year-old individual with a smooth face, as they have no beard. The combination of these features presents a distinctive and unique facial appearance." # Example from your CSV

# Generate the image
image = pipe(prompt).images[0]

# Save the image
image.save("generated_image.png")