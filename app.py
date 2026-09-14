import torch
import diffusers
import streamlit as st
from PIL import ImageDraw, ImageFont

MODEL_ID = "CompVis/stable-diffusion-v1-4"
LORA_WEIGHTS = "rschroll/maya_model_v1_lora"
LORA_FILE = "pytorch_lora_weights.safetensors"


def get_device_config():
    if torch.cuda.is_available():
        return "cuda", torch.float16
    return "cpu", torch.float32


device, dtype = get_device_config()


@st.cache_resource
def load_model():
    pipeline = diffusers.AutoPipelineForText2Image.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
    )
    pipeline.load_lora_weights(LORA_WEIGHTS, weight_name=LORA_FILE)
    pipeline.to(device)
    return pipeline


def generate_images(prompt, pipeline, n):
    return pipeline([prompt] * n).images


def add_text_to_image(
    image,
    text,
    text_color="white",
    outline_color="black",
    font_size=50,
    border_width=2,
    font_path="arial.ttf",
):
    font = ImageFont.truetype(font_path, size=font_size)
    draw = ImageDraw.Draw(image)
    width, height = image.size

    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    x = (width - text_width) / 2
    y = (height - text_height) / 2

    draw.text(
        (x, y),
        text,
        font=font,
        fill=text_color,
        stroke_width=border_width,
        stroke_fill=outline_color,
    )


def generate_memes(prompt, text, pipeline, n):
    images = generate_images(prompt, pipeline, n)
    for image in images:
        add_text_to_image(image, text)
    return images


def main():
    st.set_page_config(
        page_title="Diffusion Meme Generator",
        page_icon="🎨",
        layout="centered",
    )
    st.title("Diffusion Model Meme Generator")
    st.caption(f"Stable Diffusion + LoRA • Running on {device}")

    with st.sidebar:
        st.header("Generation")
        num_images = st.number_input(
            "Number of Images",
            min_value=1,
            max_value=10,
            value=1,
            step=1,
        )
        prompt = st.text_area("Text-to-Image Prompt")
        text = st.text_area("Text to Display")
        generate = st.button("Generate Images", type="primary")

    if generate:
        if not prompt.strip():
            st.error("Please enter a prompt.")
            return
        if not text.strip():
            st.error("Please enter the meme text.")
            return

        with st.spinner("Generating images..."):
            pipeline = load_model()
            images = generate_memes(prompt, text, pipeline, int(num_images))

        st.subheader("Generated Images")
        for image in images:
            st.image(image, use_container_width=True)


if __name__ == "__main__":
    main()
