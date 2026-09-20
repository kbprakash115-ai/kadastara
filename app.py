import streamlit as st
import requests
from PIL import Image
import io
import numpy as np
import tifffile

Image.MAX_IMAGE_PIXELS = None

st.set_page_config(page_title="Kadastara AI", layout="wide")
st.title("🛰️ Kadastara — AI Cadastral Mapping")

BACKEND_URL = "https://kbprakash115-ai--kadastara-ai-backend-predict.modal.run"

def load_image(uploaded_file):
    data = uploaded_file.getvalue()
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        pass
    arr = tifffile.imread(io.BytesIO(data))
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[0] < arr.shape[-1]:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255
        arr = arr.astype(np.uint8)
    return Image.fromarray(arr).convert("RGB")

uploaded_file = st.file_uploader(
    "Choose a drone image",
    type=["tif", "tiff", "png", "jpg", "jpeg"]
)
model_choice = st.selectbox("Model", ["buildings", "roads", "water_bodies"])

if uploaded_file is not None:
    try:
        image = load_image(uploaded_file)
    except Exception as e:
        st.error(f"Could not read image: {e}")
        st.stop()

    st.image(image, caption="Uploaded image", width="stretch")

    if st.button("🚀 Run Diagnostic"):
        # Send resized PNG (match backend preprocessing)
        small = image.resize((512, 512))
        buf = io.BytesIO()
        small.save(buf, format="PNG")
        png_bytes = buf.getvalue()

        with st.spinner("Running... (first run 1-2 min)"):
            files = {"file": ("image.png", png_bytes, "image/png")}
            params = {"model_name": model_choice}
            try:
                r = requests.post(BACKEND_URL, files=files, params=params, timeout=300)
                r.raise_for_status()
            except requests.exceptions.RequestException as e:
                st.error(f"Backend error: {e}")
                st.stop()

        # The backend returns a PNG mask
        mask = np.array(Image.open(io.BytesIO(r.content)).convert("L"))

        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Model input (resized)")
            st.image(small, width="stretch")
        with col2:
            st.subheader("Probability (raw)")
            st.image(mask, width="stretch", clamp=True)
        with col3:
            st.subheader("Mask @ 0.5")
            st.image(mask, width="stretch")

        # Histogram of the raw probabilities
        st.subheader("Probability distribution")
        hist_vals = np.histogram(mask, bins=20, range=(0, 255))[0]
        st.bar_chart(hist_vals)

        st.info(
            "If the probability image is mostly white or mostly uniform, "
            "the model isn't seeing the right input. If it's structured but "
            "mismatched to buildings, we need to change the threshold or "
            "the preprocessing."
        )
