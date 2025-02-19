#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from tabnanny import verbose
from typing import List, Optional

import click
import soundfile as sf
import torch
import tqdm
from kokoro import KPipeline


class KokoroTTS:
    def __init__(self, lang_code: Optional[str] = "a", device: Optional[str] = "cpu"):
        self.device = device
        self.main_pipeline = KPipeline(lang_code=lang_code, device=device)
        self.chunk_pipeline = KPipeline(lang_code=lang_code, model=False, device="cpu")
        pass

    def _chunk_text(self, text: str) -> List[str]:
        chunking = self.chunk_pipeline(text)
        chunks = []
        # does not produce audio
        for gs, ps, _ in chunking:
            chunks.append(gs)
        return chunks

    def text_to_speech(
        self,
        text: str,
        output_file: str,
        voice: Optional[str] = "af_heart",
        speed: Optional[float] = 1.0,
        verbose: Optional[bool] = False,
        tqdm_position: Optional[int] = 0,
    ):
        chunks = self._chunk_text(text)
        generator = self.main_pipeline(chunks, voice=voice, speed=speed)
        total_len = len(chunks)
        generated_audio = None

        t = tqdm.tqdm(generator, total=total_len, unit="chunk", position=tqdm_position)
        for i, (gs, ps, audio) in enumerate(t):
            if verbose:
                t.write("=" * 80)
                t.write(gs)  # gs => graphemes/text
                t.write("-" * 20)
                t.write(ps)  # ps => phonemes
            if generated_audio is None:
                generated_audio = audio.to("cpu")
            else:
                generated_audio = torch.cat((generated_audio, audio.to("cpu")), dim=0)
        t.write(f"Saving to {output_file}.flac")
        t.close()
        sf.write(f"{output_file}.flac", generated_audio, 24000, compression_level=1)


@click.command()
@click.option("--text", "-t", type=str, required=True, help="The text to convert to speech.")
@click.option("--output_file", "-o", type=str, required=True, help="The output file name.")
@click.option("--device", "-d", type=str, default="cpu", help="The device to use for processing.", show_default=True)
@click.option("--verbose", "-v", is_flag=True, help="Prints the phonemes and graphemes.", show_default=True)
def __main__(text: str, output_file: str, device: str):
    if os.path.exists(text):
        with open(text, "r") as f:
            text = f.read()
    tts = KokoroTTS(device=device)
    tts.text_to_speech(text, output_file, verbose=verbose)


if __name__ == "__main__":
    __main__()
