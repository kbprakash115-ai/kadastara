import streamlit as st
import requests
from PIL import Image
import io
import numpy as np

# Allow large drone images
Image.MAX_IMAGE_PIXELS = None

st.set_page_config(page_title="Kadastara AI", layout="wide")
st.title("🛰️ Kadastara — AI Cadastral Mapping")
st.write("Upload a drone image to extract buildings, roads, and water bodies.")

BACKEND_URL = "https://kbprakash115-ai--kadastara-ai-backend-predict.modal.run"


def load_image(uploaded_file):
    """Read uploaded image, including GeoTIFFs that PIL can't handle directly."""
    data = uploaded_file.getvalue()

    # Try PIL first (works for PNG, JPG, plain TIFF)
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        pass

    # Fallback: tifffile handles GeoTIFFs
    import tifffile
    arr = tifffile.imread(io.BytesIO(data))

    # Handle different band layouts
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[0] < arr.shape[-1]:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.shape[-1] == 4:
        arr = arr[..., :3]

    # Normalize to uint8 if needed
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255
        arr = arr.astype(np.uint8)

    return Image.fromarray(arr).convert("RGB")


uploaded_file = st.file_uploader(
    "Choose a drone image",
    type=["tif", "tiff", "png", "jpg", "jpeg"]
)

model_choice = st.selectbox(
    "Select model to run",
    ["buildings", "roads", "water_bodies"]
)

if uploaded_file is not None:
    try:
        image = load_image(uploaded_file)
    except Exception as e:
        st.error(f"Could not read this image: {e}")
        st.info("Try converting the file to PNG or JPG first.")
        st.stop()

    st.image(image, caption="Uploaded image", width="stretch")

    if st.button("🚀 Run AI Analysis"):
        # Convert to PNG before sending (so backend never sees the TIFF)
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        png_bytes = buf.getvalue()

        with st.spinner("Sending to AI backend... (first run can take 1-2 min)"):
            files = {"file": ("image.png", png_bytes, "image/png")}
            params = {"model_name": model_choice}

            try:
                response = requests.post(
                    BACKEND_URL, files=files, params=params, timeout=300
                )
                response.raise_for_status()

                mask_image = Image.open(io.BytesIO(response.content))
                st.subheader(f"Result: {model_choice.capitalize()}")
                st.image(mask_image, caption=f"{model_choice} mask", width="stretch")
                st.success("Analysis complete!")
            except requests.exceptions.RequestException as e:
                st.error(f"Error connecting to the AI backend: {e}")
