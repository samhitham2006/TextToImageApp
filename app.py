import streamlit as st
import torch
import os
from diffusers import StableDiffusionPipeline

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="AI Text-to-Image Generator",
    layout="wide"
)

st.title("🎨 AI Text-to-Image Generator")

# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.title("Settings")

steps = st.sidebar.slider(
    "Inference Steps",
    min_value=5,
    max_value=20,
    value=10
)

st.sidebar.markdown("---")

st.sidebar.subheader("About Project")

st.sidebar.write(
    """
    This application converts text prompts into images
    using the Stable Diffusion AI model.
    """
)

# --------------------------------------------------
# Prompt History
# --------------------------------------------------

if "history" not in st.session_state:
    st.session_state.history = []

# --------------------------------------------------
# Load Model
# --------------------------------------------------

@st.cache_resource
def load_model():

    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float32
    )

    pipe = pipe.to("cpu")

    return pipe

pipe = load_model()

# --------------------------------------------------
# User Input
# --------------------------------------------------

prompt = st.text_area(
    "Enter Image Prompt",
    value="A beautiful sunset over mountains",
    height=120
)

col1, col2 = st.columns(2)

with col1:
    generate = st.button("Generate Image")

with col2:
    clear = st.button("Clear History")

# --------------------------------------------------
# Clear History
# --------------------------------------------------

if clear:
    st.session_state.history = []
    st.rerun()

# --------------------------------------------------
# Generate Image
# --------------------------------------------------

if generate and prompt:

    with st.spinner("Generating image... Please wait."):

        try:

            image = pipe(
                prompt,
                num_inference_steps=steps,
                guidance_scale=7.5,
                width=512,
                height=512
            ).images[0]

        except Exception as e:

            st.error(f"Generation failed: {e}")
            st.stop()

        st.session_state.history.append(prompt)

        st.success("Image Generated Successfully!")

        st.image(
            image,
            caption=prompt,
            use_container_width=True
        )

        st.info(
            f"""
Prompt: {prompt}

Inference Steps: {steps}

Image Size: 512x512
"""
        )

        os.makedirs("generated_images", exist_ok=True)

        filename = (
            f"generated_images/image_"
            f"{len(st.session_state.history)}.png"
        )

        image.save(filename)

        with open(filename, "rb") as file:

            st.download_button(
                label="⬇ Download Image",
                data=file,
                file_name=os.path.basename(filename),
                mime="image/png"
            )

# --------------------------------------------------
# Prompt History
# --------------------------------------------------

if st.session_state.history:

    st.markdown("---")
    st.subheader("Prompt History")

    for i, item in enumerate(
            reversed(st.session_state.history), 1):

        st.write(f"{i}. {item}")

# --------------------------------------------------
# Workflow Section
# --------------------------------------------------

st.markdown("---")

st.subheader("Project Workflow")

st.markdown("""
1. User enters a text prompt.
2. Stable Diffusion processes the prompt.
3. AI generates an image.
4. Generated image is displayed.
5. Image is automatically saved.
6. User can download the image.
""")

# --------------------------------------------------
# Footer
# --------------------------------------------------

st.markdown("---")

st.caption(
    "Mini Project: Text-to-Image Application using "
    "Streamlit and Stable Diffusion"
)