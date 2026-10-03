# Diffusion Meme Generator

A minimal Streamlit portfolio project that combines **Stable Diffusion + LoRA** with Pillow text rendering to turn a text prompt into a generated meme image.

## What it demonstrates

- Using a pretrained diffusion pipeline for text-to-image generation
- Loading LoRA weights into the pipeline
- CUDA detection with CPU fallback
- Reproducible generation through an explicit seed
- Simple image post-processing with Pillow
- A small interactive ML application built with Streamlit

## How it works

```text
Prompt + Meme Text + Seed
            |
            v
       Streamlit UI
            |
            v
  Stable Diffusion + LoRA
            |
            v
      Generated Image
            |
            v
       Pillow Overlay
            |
            v
        Meme Image
```

## Run locally

Python 3.11+ is recommended.

```bash
git clone https://github.com/alim8rby/Diffusion_Meme_Generator.git
cd Diffusion_Meme_Generator

python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

The first generation downloads the Stable Diffusion model and LoRA weights from Hugging Face, so the initial run can take longer and requires network access.

## Using the app

1. Enter a text-to-image prompt, for example:
   `a confused astronaut looking at a broken computer, cinematic lighting`
2. Enter the text you want overlaid on the generated image.
3. Choose the number of images.
4. Keep the default seed or change it to explore different outputs.
5. Click **Generate Images**.

The same prompt and seed are intended to make generation reproducible under the same software/model environment.

## Model

- Base model: `CompVis/stable-diffusion-v1-4`
- LoRA: `rschroll/maya_model_v1_lora`
- LoRA file: `pytorch_lora_weights.safetensors`

Model loading and inference are handled through Hugging Face Diffusers.

## Hardware

CUDA is used automatically when available. CPU inference is supported, but diffusion generation can be substantially slower.

## Project structure

```text
app.py                  # Streamlit UI and core generation pipeline
requirements.txt        # Python dependencies
tests/                  # Lightweight unit tests
docs/                   # Short project documentation
arial.ttf               # Font used for meme text
```

## Testing

The test suite focuses on the non-model logic so it can run without downloading a diffusion model:

```bash
python -m unittest discover -s tests -v
```

CI also performs a Python compilation check on pushes and pull requests.

## Limitations

This is intentionally a small portfolio application, not a production image-generation service. It does not include model fine-tuning, a database, user accounts, or a hosted inference backend.

## Portfolio focus

The goal of this project is to demonstrate a clear end-to-end generative-AI pipeline with a small amount of practical engineering around reproducibility, validation, testing, and an interactive interface.
