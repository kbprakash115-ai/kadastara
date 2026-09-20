import streamlit as st
import requests
from PIL import Image
import io

st.set_page_config(page_title="Kadastara AI", layout="wide")
st.title("🛰️ Kadastara — AI Cadastral Mapping")
st.write("Upload a drone image to extract buildings, roads, and water bodies.")

# Your live Modal backend URL
BACKEND_URL = "https://kbprakash115-ai--kadastara-ai-backend-predict.modal.run"

uploaded_file = st.file_uploader(
    "Choose a drone image",
    type=["tif", "tiff", "png", "jpg", "jpeg"]
)

model_choice = st.selectbox(
    "Select model to run",
    ["buildings", "roads", "water_bodies"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    if st.button("🚀 Run AI Analysis"):
        with st.spinner("Sending to AI backend..."):
            files = {
                "file": (uploaded_file.name, uploaded_file.getvalue(), "image/png")
            }
            params = {"model_name": model_choice}

            try:
                response = requests.post(
                    BACKEND_URL, files=files, params=params, timeout=180
                )
                response.raise_for_status()

                mask_image = Image.open(io.BytesIO(response.content))
                st.subheader(f"Result: {model_choice.capitalize()}")
                st.image(mask_image, caption=f"{model_choice} mask", use_container_width=True)
                st.success("Analysis complete!")
            except requests.exceptions.RequestException as e:
                st.error(f"Error connecting to the AI backend: {e}")
