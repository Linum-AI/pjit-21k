<div align="center">

# A Text-to-Image Model Trained for $21K<br>(Pyramid-JiT)

**[Sahil Chopra](https://thesahilchopra.com/) · [Manu Chopra](https://www.linkedin.com/in/manu-chopra-50360b170/)**

*Linum*

[![Blog post](https://img.shields.io/badge/blog%20post-linum.ai%2Ffield--notes%2Fpjit--21k-6b4fd8)](https://www.linum.ai/field-notes/pjit-21k)
[![Weights](https://img.shields.io/badge/%F0%9F%A4%97%20weights-Linum--AI%2Fpjit--21k-ffcc4d)](https://huggingface.co/Linum-AI/pjit-21k)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

</div>

![Five P-JiT 21K samples: a close-up portrait of an elderly woman in a cream headscarf, a man blowing out birthday candles, a children's book illustration of a bear sharing honey with a rabbit, a giant panda in a bamboo grove, and an alpine meadow of wildflowers](assets/samples.jpg)

*Five samples from the [blog post](https://www.linum.ai/field-notes/pjit-21k)'s gallery, at the
model's native ~1 MP sizes. Their prompts and seeds are [below](#reproducing-the-samples).*

> [!WARNING]
> **Research checkpoint: this is an experimental research artifact, not a production model.**
> P-JiT 21K is a 2B-parameter model trained on 660M samples for about $21K of compute, as a
> stepping stone towards our next video model. We're releasing it because the results
> surprised us for a model this size, and to encourage others to explore efficient training
> methods like ours. You should expect rough edges.

> [!IMPORTANT]
> **This model cannot write text or generate posters.** It was trained on little to no images
> of rendered text: no infographics, PDFs, pages of books, slides or posters. We filtered that
> data out on purpose (it isn't useful for our goal of animation tools), so prompts that ask
> for words, signs, logos, title cards or posters will not work. The prompt upsampler returns a
> `text_requested` error for these prompts instead of a caption.

P-JiT 21K is the [P-JiT](https://www.linum.ai/field-notes/pyramid-jit) architecture trained for longer:
610M samples at 512px, 50M at 1024px, then post-training with a variant of
[Finite Difference Flow Optimization (FDFO)](https://arxiv.org/abs/2603.12893). The
[blog post](https://www.linum.ai/field-notes/pjit-21k) covers the training run.

## Getting started

```bash
sudo apt-get install -y build-essential python3-dev python3-venv   # Triton compiles small CUDA shims at first use
git clone https://github.com/Linum-AI/pjit-21k && cd pjit-21k
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

Generate the portrait at the top left (no API key needed: the prompt is already a detailed
caption, so it skips the upsampler):

```bash
PROMPT="Close-up portrait of an elderly Middle Eastern woman looking calmly into the lens, her face filling the tall frame. A cream headscarf and the collar of a dark green cardigan frame her chin. Soft overcast daylight smooths the fine lines of her skin, and the olive grove behind melts into silvery blur."
python generate.py --no_upsample --aspect_ratio tall --seeds 2 --prompt "$PROMPT" --out_dir outputs
# writes outputs/seed_2.png
```

The weights and the Qwen3.5-4B text encoder download from the Hugging Face Hub on first use.
The model needs a CUDA GPU: its weights are 8.7 GB in fp32 and the text encoder 8 GB in bf16,
plus activations.

### Your own prompts

The model was trained on dense captions of 50 to 80 words, and short prompts give noticeably
worse images. So by default `generate.py` first rewrites your prompt into a caption of that
kind with Claude Opus 5.5 through [OpenRouter](https://openrouter.ai), which needs an
OpenRouter API key ([get one here](https://openrouter.ai/keys)):

```bash
export OPENROUTER_API_KEY=...
python generate.py --prompt "oil painting of weathered fisherman smoking a pipe" \
    --aspect_ratio square --seeds 0 --out_dir outputs
```

The upsampler turns that prompt into a caption like this one (Claude writes a slightly
different one each time):

> An oil painting of a weathered fisherman in his seventies in a close head-and-shoulders
> portrait, a curved wooden pipe held in the corner of his mouth and a thin curl of smoke rising
> beside his face. His skin is deeply lined and ruddy, his white beard rough, and he wears a dark
> knitted cap and a yellow oilskin jacket. Warm lamplight falls on one cheek, laid on in thick,
> textured brushstrokes against a muted blue-grey background.

The caption the model actually saw is printed and saved to `outputs/prompt.json` beside the
PNGs. Pass `--no_upsample` to send a caption you've written yourself as is (no key needed).

### Aspect ratios

`--aspect_ratio` picks one of the five ~1 MP sizes the model was trained at:

| `--aspect_ratio` | Width x height |
|---|---|
| `square` (default) | 1024 x 1024 |
| `tall` | 864 x 1216 |
| `4:3` | 1152 x 864 |
| `3:2` | 1248 x 832 |
| `16:9` | 1280 x 720 |

`--width` and `--height` set any other size (sides that aren't a multiple of 32 are padded and
cropped back), but the model has only seen these five.

## Reproducing the samples

Each image above is one seed, sampled with the default settings, no upsampling (the prompts are
already captions) and the initial noise drawn as on an H100 SXM. The portrait is the
[Getting started](#getting-started) command; the other four are below. The gallery images were
sampled five seeds to a batch, and the batch size changes how the GPU rounds its matrix
multiplies, so one seed on its own gives the same image with pixel values that differ by
rounding (under 1/255 on average).

**Birthday** (seed 0):

```bash
PROMPT="Cheeks puffed, a white man in his sixties leans over a birthday cake on a kitchen table and blows out a ring of thin candles, the flames bending away from him. A checked flannel shirt covers his grey t-shirt, his grey hair neatly combed. The cake has plain white frosting and fresh strawberries, and medium-format film keeps the colours warm and natural in the centred square composition."
python generate.py --no_upsample --aspect_ratio square --seeds 0 --prompt "$PROMPT"
```

**Bear and rabbit** (seed 4):

```bash
PROMPT="A children's book illustration of a big brown bear sharing a jar of honey with a small white rabbit, the bear holding the jar out toward the rabbit as they sit together under a tree, drawn in soft colored pencil with warm, gentle colors. The pair sits in the center of the frame, seen from the front at eye level, with soft, grainy pencil strokes and a pale, simply sketched background that keeps the focus on their friendly exchange."
python generate.py --no_upsample --aspect_ratio square --seeds 4 --prompt "$PROMPT"
```

**Panda** (seed 1):

```bash
PROMPT="Golden-hour light slants through a bamboo grove onto a giant panda sitting on mossy ground, a wide cinematic view with the bear placed left of centre. It grips a green stalk in both front paws and chews, its black-and-white fur glowing at the edges where the low sun catches it. Tall bamboo stems stretch away to the right, their leaves lit amber, and soft haze fills the gaps between them."
python generate.py --no_upsample --aspect_ratio 16:9 --seeds 1 --prompt "$PROMPT"
```

**Meadow** (seed 1):

```bash
PROMPT="An alpine meadow sweeps up toward jagged grey peaks, thick with wildflowers: purple lupines, yellow arnica, white daisies and pink clover crowding the foreground. Soft overcast light keeps the colours rich and even, and low cloud drapes the higher ridges in pale wisps. A thin stream winds through the grass from the lower right, its banks lined with moss, and a scattering of dark spruce trees marks the far edge of the meadow where the slope turns to scree."
python generate.py --no_upsample --aspect_ratio 3:2 --seeds 1 --prompt "$PROMPT"
```

## Limitations

This is a small model (a 2B-parameter DiT with a 4B text encoder), and there are areas where it
struggles:

- Full-body shots
- Many entities (six cars, crowds of people)
- Interactions between several people
- World knowledge (six strings on a guitar, the dots on a die)

## Architecture

| | |
|---|---|
| Input patch | 32x32 px, linear 3,072 -> 1,024 -> 2,944 (1,024 tokens at 1024²) |
| Trunk | 22 single-stream DiT blocks · width 2,944 · 23 heads x 128 · SwiGLU 7,936 |
| Refiners | 2 image + 2 text blocks before the trunk, timestep-modulated |
| Block | Sandwich RMSNorm, low-rank AdaLN (scale-in, tanh gate-out), q/k RMSNorm, sigmoid attention gate |
| Position | 3-D RoPE on image tokens; height and width positions x 0.5 at 1024px (Position Interpolation from the 512px stage) |
| Text | Qwen3.5-4B (frozen), hidden layers 7, 15, 27 concatenated -> MLP · 256 tokens |
| Parameters | 2.18B |

The training-time readout heads and the PixelREPA adapter play no part in sampling and are not
included. The [pyramid-jit](https://github.com/Linum-AI/pyramid-jit) repository has reference
implementations of them and of the rest of the pretraining objective.

## Sampling

50 Euler steps from noise (scale 4, the 1024px noise level) to the image, with
[adaptive projected guidance](https://arxiv.org/abs/2410.02416) at scale 15 against an empty
negative prompt, under bf16 autocast. The state is carried in fp32 and the model sees it
rounded to bf16, as in the sampler the release images were made with.

The initial noise is drawn exactly as `torch.randn` draws it on an H100 SXM, the GPU the model
was trained on ([`pjit_21k/noise.py`](pjit_21k/noise.py)), so a seed gives the same image on
any NVIDIA GPU rather than one that depends on the GPU model. Pass `--native_noise` to use the
local GPU's own draw. Attention runs on FlashAttention-3 when `flash_attn_interface` is
installed and on PyTorch SDPA otherwise.

## Citation

```bibtex
@online{chopra2026pjit21k,
  title = {Training a Text-to-Image Model for \$21K},
  author = {Chopra, Sahil and Chopra, Manu},
  year = {2026},
  url = {https://www.linum.ai/field-notes/pjit-21k}
}
```

## Authorship

This repository was written by Claude (Anthropic's Opus 5.5 model, running in Claude Code).
Linum asked it to extract the model and inference code from Linum's internal experiment
repository and to add the prompt upsampler. The model, the training, and the review of this
repository are Linum's.

## About Linum

Linum is a team of two brothers building a tiny-yet-powerful AI research lab. We train our
own generative media models from scratch.

**Subscribe to [Field Notes](https://buttondown.com/linum-ai)**: technical deep dives on
building generative video models from the ground up, plus updates on new releases from Linum.

**Contact:** hello@linum.ai

## License

Apache-2.0. Copyright 2026 Linum Inc. See [LICENSE](LICENSE).
