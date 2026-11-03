# Copyright 2026 Linum Inc. Licensed under the Apache License, Version 2.0.

"""
File: generate.py
Description: Command-line text-to-image generation with P-JiT 21K. The prompt is first
    upsampled into a dense caption by Claude Opus 5.5 through OpenRouter (needs
    OPENROUTER_API_KEY), unless --no_upsample is given. The upsampler's response (the prompt,
    the upsampled prompt and an error) is printed as JSON; on an error ("illicit_content" or
    "text_requested") the upsampled prompt is empty and the script exits with status 1
    without generating. Otherwise it writes one PNG per seed plus a prompt.json with the
    response, the caption the model saw, and the settings.

    export OPENROUTER_API_KEY=...
    python generate.py --prompt "a lighthouse in a storm" --aspect_ratio tall --seeds 0,1,2

    python generate.py --no_upsample --prompt "$DETAILED_CAPTION" --seeds 0

    With no --prompt, the caption of the README's portrait sample is used; with
    --no_upsample --aspect_ratio tall --seeds 2, seed_2.png is that sample.
"""

import argparse
import json
import os
from typing import List

import torch
from PIL import Image

from pjit_21k import (
    ASPECT_RATIOS, PyramidJiT, QwenTextEncoder, SamplerConfig, active_attention_backend,
    generate, set_attention_backend, upsample_prompt,
)
from pjit_21k.attention import BACKENDS
from pjit_21k.upsampler import (
    API_KEY_ENV, DEFAULT_MODEL, DEFAULT_REASONING_EFFORT, ERROR_ILLICIT_CONTENT,
    ERROR_TEXT_REQUESTED, UpsampleResult, resolve_api_key,
)

ERROR_MESSAGES = {
    ERROR_ILLICIT_CONTENT: "the prompt names a real person or asks for explicit content",
    ERROR_TEXT_REQUESTED: "the prompt asks for writing in the image, which this model "
                          "cannot draw; describe the scene without the text",
}

DEFAULT_WEIGHTS = "Linum-AI/pjit-21k"
DEFAULT_QWEN = "Qwen/Qwen3.5-4B"
# The README's portrait sample (tall, seed 2).
DEFAULT_PROMPT = (
    "Close-up portrait of an elderly Middle Eastern woman looking calmly into the lens, her "
    "face filling the tall frame. A cream headscarf and the collar of a dark green cardigan "
    "frame her chin. Soft overcast daylight smooths the fine lines of her skin, and the olive "
    "grove behind melts into silvery blur.")


# --------------------------------
# CLI
# --------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    defaults = SamplerConfig()
    parser = argparse.ArgumentParser(description="Generate images with P-JiT 21K")
    parser.add_argument(
        "--prompt", type=str, default=DEFAULT_PROMPT,
        help="Prompt or caption. Default: the README's portrait caption")
    parser.add_argument(
        "--no_upsample", action="store_true",
        help="Use the prompt as written instead of upsampling it into a dense caption "
             "(no OpenRouter key needed). Only do this with a detailed caption of ~50-80 words")
    parser.add_argument(
        "--openrouter_api_key", type=str, default=None,
        help=f"OpenRouter API key for the upsampler. Default: ${API_KEY_ENV}")
    parser.add_argument(
        "--upsampler_model", type=str, default=DEFAULT_MODEL, help="OpenRouter model id")
    parser.add_argument(
        "--upsampler_reasoning_effort", type=str, default=DEFAULT_REASONING_EFFORT,
        choices=("low", "medium", "high"))
    parser.add_argument(
        "--weights", type=str, default=DEFAULT_WEIGHTS,
        help="Hugging Face Hub id, or a local directory with model.safetensors + config.json")
    parser.add_argument(
        "--qwen_model_path", type=str, default=DEFAULT_QWEN,
        help="Local directory or Hugging Face id of Qwen3.5-4B")
    parser.add_argument(
        "--negative_prompt", type=str, default=defaults.negative_prompt,
        help="Caption for the unconditional branch (default: empty)")
    parser.add_argument("--seeds", type=str, default="0", help="Comma-separated seeds")
    parser.add_argument(
        "--aspect_ratio", type=str, default="square", choices=sorted(ASPECT_RATIOS),
        help="One of the five ~1 MP sizes the model was trained at: square 1024x1024, "
             "tall 864x1216, 4:3 1152x864, 3:2 1248x832, 16:9 1280x720 (width x height)")
    parser.add_argument(
        "--height", type=int, default=None, help="Overrides --aspect_ratio (with --width)")
    parser.add_argument(
        "--width", type=int, default=None, help="Overrides --aspect_ratio (with --height)")
    parser.add_argument("--sampling_steps", type=int, default=defaults.sampling_steps)
    parser.add_argument("--guidance_scale", type=float, default=defaults.guidance_scale)
    parser.add_argument(
        "--one_seed_per_call", action="store_true",
        help="Sample each seed in its own batch instead of all seeds together. A seed's "
             "noise is the same either way, but GEMM tiling depends on the batch size, so "
             "the pixels differ slightly")
    parser.add_argument(
        "--native_noise", action="store_true",
        help="Draw the initial noise with this GPU's own torch.randn instead of reproducing "
             "the H100 SXM draw (seeds then give different images on different GPU models)")
    parser.add_argument(
        "--attention_backend", type=str, default="auto", choices=BACKENDS,
        help="auto: FlashAttention-3 if installed, else PyTorch SDPA")
    parser.add_argument("--out_dir", type=str, default="outputs", help="Where to write PNGs")
    args = parser.parse_args()
    if (args.height is None) != (args.width is None):
        parser.error("--height and --width go together")
    return args


def main() -> None:
    """Upsample the prompt (unless disabled) and generate images."""
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    height, width = ((args.height, args.width) if args.height is not None
                     else ASPECT_RATIOS[args.aspect_ratio])

    # Upsample before loading anything, so a missing key, a failed request or an upsampler
    # error stops the run before the model loads.
    if args.no_upsample:
        response = UpsampleResult(prompt=args.prompt, upsampled_prompt="", error=None)
        caption = args.prompt
    else:
        api_key = resolve_api_key(api_key=args.openrouter_api_key)
        print(f"upsampling with {args.upsampler_model} ({args.upsampler_reasoning_effort} "
              "reasoning)...")
        response = upsample_prompt(
            prompt=args.prompt, height=height, width=width, api_key=api_key,
            model=args.upsampler_model, reasoning_effort=args.upsampler_reasoning_effort)
        print(json.dumps(response.to_dict(), indent=2))
        if response.error:
            raise SystemExit(f"upsampler error: {response.error} "
                             f"({ERROR_MESSAGES[response.error]}). No images were generated.")
        caption = response.upsampled_prompt

    sampler = SamplerConfig(
        height=height,
        width=width,
        sampling_steps=args.sampling_steps,
        guidance_scale=args.guidance_scale,
        negative_prompt=args.negative_prompt,
        noise_sm_count=None if args.native_noise else SamplerConfig().noise_sm_count,
    )

    set_attention_backend(backend=args.attention_backend)
    print(f"attention backend: {active_attention_backend()}")
    model = PyramidJiT.from_pretrained(weights_dir=args.weights, device="cuda")
    text_encoder = QwenTextEncoder(
        model_path=args.qwen_model_path,
        extraction_layers=model.config.text_layers,
        max_length=model.config.text_len)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "prompt.json"), "w") as fh:
        json.dump({
            **response.to_dict(),
            "caption": caption,
            "upsampled": not args.no_upsample,
            "upsampler_model": None if args.no_upsample else args.upsampler_model,
            "seeds": seeds,
            "sampler": sampler.to_dict(),
        }, fh, indent=2)

    seed_groups: List[List[int]] = [[s] for s in seeds] if args.one_seed_per_call else [seeds]
    for group in seed_groups:
        images = generate(
            model=model, text_encoder=text_encoder, prompt=caption, seeds=group,
            sampler=sampler)
        for seed, image in zip(group, images):
            path = os.path.join(args.out_dir, f"seed_{seed}.png")
            save_png(image=image, path=path)
            print(f"wrote {path}")


# --------------------------------
# OUTPUT
# --------------------------------

def save_png(image: torch.Tensor, path: str) -> None:
    """
    Write a (3, 1, H, W) uint8 tensor as a PNG.

    Args:
        image (torch.Tensor):
            uint8 image tensor.
        path (str):
            Destination path.
    """
    array = image[:, 0].permute(1, 2, 0).cpu().numpy()
    Image.fromarray(array).save(path)


if __name__ == "__main__":
    main()
