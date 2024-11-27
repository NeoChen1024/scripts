#!/usr/bin/env python
# Simple Python script to generate image captions using OpenAI API
# Designed to be used with GNU parallel to achieve batch inference

import os
import sys
import PIL
import base64
from PIL import Image
import requests
import json
from io import BytesIO
from openai import OpenAI
import argparse
from rich import print
from rich.padding import Padding

def none_or_str(value):
	if value is None:
		return "None"
	return str(value)

def parse_args():
    parser = argparse.ArgumentParser(description='Generate image captions using Mistral API')
    parser.add_argument('-i', '--image_file', required=True, help='Path to the input image file')
    parser.add_argument('-o', '--caption_output', required=False, help='Path to save the caption output (default: same as image file with .txt extension)')
    parser.add_argument('-vp', '--vision_prompt_file', required=False, help='Vision prompt file for caption generation (default: built-in)')
    parser.add_argument('-ap', '--additional_prompt', required=False, help='Additional prompt for caption generation, replace %% in the prompt, " %%" will be removed if it\'s not set (default: None)')
    parser.add_argument('-m', '--model', required=False, help='Model to use for caption generation (default: pixtral-large-latest if LLM_MODEL_NAME environment variable is not set)')
    parser.add_argument('-k', '--api_key', required=False, help='Override LLM API key (default: content of LLM_API_KEY environment variable)')
    parser.add_argument('-u', '--base_url', required=False, help='Override LLM base URL (default: Mistral\'s)')
    parser.add_argument('-t', '--temperature', required=False, type=float, default=0.7, help='Temperature for caption generation (default: 0.7)')
    parser.add_argument('-v', '--verbose', required=False, default=False, help='Verbose output (default: False)', action='store_true')
    parser.add_argument('-ex', '--existing_caption', required=False, default="overwrite", choices=["overwrite", "skip"], help='"overwrite" or "skip" existing caption file (default: overwrite)')
    return parser.parse_args()

args = parse_args()
image_path = args.image_file
additional_prompt = args.additional_prompt
verbose = args.verbose
temperature = args.temperature

if verbose:
	print("================================================")
	print("Captioning [yellow]" + image_path + "[/yellow]")
	print("================================================")
else:
	print(">> [yellow]" + image_path + "[/yellow]")

if args.caption_output is None:
	caption_output = os.path.splitext(image_path)[0] + ".txt"
else:
	caption_output = args.caption_output

if args.existing_caption == "skip" and os.path.exists(caption_output):
	print("Caption file already exists, skipping...")
	sys.exit(0)

if verbose:
	print("Caption output:\t\t[yellow]" + caption_output + "[/yellow]")
	print("Additional prompt:\t[green]" + none_or_str(additional_prompt) + "[/green]")

# Check if the API key is set
api_key = "xxxx" # placeholder, it's fine for it to be any string for local LLM
if (key := os.environ.get("LLM_API_KEY")) is not None:
	api_key = key
# Override if provided by command line argument
if args.api_key is not None:
	api_key = args.api_key

# Get model name from environment variable if it exists
model_name = "pixtral-large-latest" # default model from Mistral
if (name := os.environ.get("LLM_MODEL_NAME")) is not None:
	model_name = name
# Override if provided by command line argument
if args.model is not None:
	model_name = args.model

if verbose:
	print("Model:\t\t\t[cyan]" + model_name + "[/cyan]")

base_url = "https://api.mistral.ai/v1"
if (url := os.environ.get("LLM_BASE_URL")) is not None:
	base_url = url
# Override if provided by command line argument
if args.base_url is not None:
	base_url = args.base_url

if verbose:
	print("Base URL:\t\t[cyan]" + none_or_str(base_url) + "[/cyan]")

vision_prompt = \
	"Describe this image, which will be used to train an image generator in natural language." \
	+ " The caption should be as descriptive, detailed, and specific as possible," \
	+ " including people's ethnicity, gender, face, eye color, hair color," \
	+ " clothing, accessories, objects, actions, and context, camera angle, etc." \
	+ " The caption should not start with phrases like \"An image of\" or \"A photo of\". %%"

if args.vision_prompt_file is not None:
	with open(args.vision_prompt_file, "r", encoding="utf-8") as f:
		vision_prompt = f.read()

if additional_prompt is not None:
	vision_prompt = vision_prompt.replace("%%", additional_prompt, 1)
else:
	vision_prompt = vision_prompt.replace(" %%", "", 1) # Remove the extra space

if verbose:
	print("Full vision prompt:")
	print(Padding("[bright_magenta]" + vision_prompt + "[/bright_magenta]", (0, 0, 0, 4)))


# Check if the image file exists
if not os.path.exists(image_path):
	print("Error: Image file does not exist.")
	sys.exit(1)

# Load the image
image = Image.open(image_path)
if image is None:
	print("Error: Unable to open image file.")
	sys.exit(1)

image_jpeg = BytesIO()

# Resize the image to fit within 2048x2048 pixels
width, height = image.size
if width > 2048 or height > 2048:
	scale_factor = max(width, height) / 2048
	image = image.resize((int(width / scale_factor), int(height / scale_factor)))

image.save(image_jpeg, format="JPEG", quality=90) # Use JPEG quality 90 to reduce file size
del image

# Convert the image to a base64 string
image_base64 = base64.b64encode(image_jpeg.getvalue()).decode("utf-8")
del image_jpeg

print("Caption:")

client = OpenAI(
    api_key=api_key,
    base_url=base_url
)

try:
	chat_response = client.chat.completions.create(
	model= model_name,
	stream=False,
	max_tokens=500,
	temperature=temperature,
		top_p=0.9,
		messages=[{
			"role": "user",
			"content": [
				{
					"type": "text",
					"text": vision_prompt
				},
				{
					"type": "image_url",
					"image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
				}
			]
		}]
	)
	caption_response = chat_response.choices[0].message.content
except Exception as e:
	print("[red]Error:\t" + str(e) + "[/red]")
	sys.exit(1)

print(Padding("[green]" + caption_response + "[/green]", (0, 0, 0, 4)))

# Save the caption to a file
caption_file = open(caption_output, "w", encoding="utf-8")
caption_file.write(caption_response + "\n")
if verbose:
	print("Caption saved to [yellow]" + caption_output + "[/yellow]")
