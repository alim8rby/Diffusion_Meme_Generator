# Diffusion Meme Generator

A Streamlit application that combines **Stable Diffusion** with **LoRA** weights to generate images from text prompts and turn them into simple memes with custom overlay text.

## Highlights

- Text-to-image generation with Stable Diffusion
- LoRA-based model customization
- Configurable batch generation
- Automatic meme text overlay
- CUDA acceleration when available, with CPU fallback
- Streamlit interface for interactive generation

## How It Works

```text
Prompt + Meme Text
        │
        ▼
   Streamlit UI
        │
        ▼
 Stable Diffusion
        │
        ▼
    LoRA Weights
        │
        ▼
 Generated Images
        │
        ▼
   Text Overlay
```

## Run Locally

```bash
git clone https://github.com/alim8rby/st-meme-app.git
cd st-meme-app
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The first generation may take longer because the diffusion model needs to be downloaded and loaded into memory.

## Technical Stack

Python · Streamlit · PyTorch · Diffusers · Transformers · Accelerate · Safetensors · Pillow

## Project Context

This project was developed in association with WorldQuant's Applied AI Lab as an applied experiment in generative AI, model adaptation, and interactive ML applications.

## Notes

The application is an experimental portfolio project rather than a production deployment. GPU hardware is recommended for practical generation speed.
