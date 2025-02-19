#!/usr/bin/env python

# CLIP Image-Text Similarity

import argparse
import os
from pprint import pp

import PIL
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


def argparser():
    parser = argparse.ArgumentParser(description="CLIP Image-Text Similarity")
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="apple/DFN5B-CLIP-ViT-H-14-378",
        help="Model ID",
    )
    parser.add_argument(
        "--images",
        "-i",
        type=str,
        required=True,
        nargs="+",
        default=[],
        help="Path to image file",
    )
    parser.add_argument(
        "--text",
        "-t",
        type=str,
        required=True,
        help="Text caption to compare",
    )
    args = parser.parse_args()
    return args


def __main__():
    args = argparser()
    print(args)
    # Use CLIP-G
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = CLIPProcessor.from_pretrained(args.model, device_map=device)
    model = CLIPModel.from_pretrained(args.model, device_map=device)
    print("Model loaded: ", args.model)
    # Load images
    images = []
    for img in args.images:
        with Image.open(img) as i:
            i = i.convert("RGB")
            images.append(i)

    print(f"{len(images)} images loaded")
    print("Compare with text: ", args.text)

    # Preprocess images and texts
    inputs = processor(
        text=args.text, images=images, return_tensors="pt", truncation=True
    ).to(device)
    print("Preprocessed")

    # Calculate similarity
    with torch.no_grad():
        output = model(**inputs)
    logits = output.logits_per_image.cpu().numpy()
    # Apply softmax
    similaritys = torch.nn.functional.softmax(torch.tensor(logits), dim=0)
    # Print similarity
    for i, val in enumerate(similaritys):
        val = val.item()
        print(f"{args.images[i]} - Text similarity: {val}")
    return


if __name__ == "__main__":
    __main__()
