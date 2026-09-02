#!/usr/bin/env python

# CLIP Image-Text Similarity

import click
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


@click.command()
@click.option(
    "--model",
    "-m",
    default="apple/DFN5B-CLIP-ViT-H-14-378",
    show_default=True,
    help="Model ID",
)
@click.option(
    "--images",
    "-i",
    required=True,
    multiple=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to an image file. Repeat for multiple images.",
)
@click.option(
    "--text",
    "-t",
    required=True,
    help="Text caption to compare",
)
def __main__(model, images, text):
    """Calculate CLIP image-text similarity."""

    # Use CLIP-G
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = CLIPProcessor.from_pretrained(model, device_map=device)
    clip_model = CLIPModel.from_pretrained(model, device_map=device)
    print("Model loaded: ", model)
    # Load images
    loaded_images = []
    for img in images:
        with Image.open(img) as i:
            i = i.convert("RGB")
            loaded_images.append(i)

    print(f"{len(loaded_images)} images loaded")
    print("Compare with text: ", text)

    # Preprocess images and texts
    inputs = processor(text=text, images=loaded_images, return_tensors="pt", truncation=True).to(device)
    print("Preprocessed")

    # Calculate similarity
    with torch.no_grad():
        output = clip_model(**inputs)
    logits = output.logits_per_image.cpu().numpy()
    # Apply softmax
    similaritys = torch.nn.functional.softmax(torch.tensor(logits), dim=0)
    # Print similarity
    for i, val in enumerate(similaritys):
        val = val.item()
        print(f"{images[i]} - Text similarity: {val}")
    return


if __name__ == "__main__":
    __main__()
