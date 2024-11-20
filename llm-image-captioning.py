#!/usr/bin/env python
# Simple Python script to generate image captions using local OpenAI API

import os
import sys
import PIL
import base64
from PIL import Image
import requests
import json
from io import BytesIO
from openai import OpenAI

if len(sys.argv) < 3:
	print("Usage: llm-image-captioning.py <image_file> <caption_output>")
	sys.exit(1)

# get parameters from command line
image_path = sys.argv[1]
caption_output = sys.argv[2]

# Runtime variables
model_name = None

# Check if the OpenAI API key is set
openai_api_key = "xxxx"
if os.environ.get("OPENAI_API_KEY") is not None:
	openai_api_key = os.environ.get("OPENAI_API_KEY")

# Set the OpenAI API key
client = OpenAI(
  api_key=openai_api_key,
  base_url="http://localhost:5000/v1"
)

# Get the list of models in dict format
models_list =  client.models.list().model_dump()

# Get the name of the model
for model in models_list['data']:
	model_name = model['id']

if model_name is None:
	print("Error: Unable to find a working model.")
	sys.exit(1)

# Check if the image file exists
if not os.path.exists(image_path):
	print("Error: Image file does not exist.")
	sys.exit(1)

# Load the image
image = Image.open(image_path)
if image is None:
	print("Error: Unable to open image file.")
	sys.exit(1)
if image.mode != "RGB":
	image = image.convert("RGB")
image_png = BytesIO()
image.save(image_png, format="PNG")

# Convert the image to a base64 string
image_base64 = base64.b64encode(image_png.getvalue()).decode("utf-8")
image_base64 = "data:image/png;base64," + image_base64

vision_prompt = "Generate an image caption that will be used to train an image generator in natural language." \
	+ " The caption should be as descriptive as possible, providing a detailed description" \
	+ " of the image content, including objects, actions, and context." \
	+ " The caption should not start with phrases like 'An image of' or 'A photo of'."

response = client.chat.completions.create(
	stream=False,
	model=model_name,
	max_tokens=250,
	messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": vision_prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"{image_base64}"},
                },
            ],
        },
    ],
)

caption_response = response.choices[0].message.content
print(">> " + image_path + "\n" + caption_response)

# Save the caption to a file
caption_file = open(caption_output, "w")
caption_file.write(caption_response + "\n")
