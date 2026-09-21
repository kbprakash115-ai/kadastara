import streamlit as st
import requests
from PIL import Image
import io
import numpy as np
import tifffile

Image.MAX_IMAGE_PIXELS = None

st.set_page_config(page_title="Kadastara AI", layout="wide")
st.title("🛰️ Kadastara — AI Cadastral Mapping")
st.write("Upload a drone image to extract buildings, roads, and water bodies.")

BACKEND_URL = "https://kbprakash115-ai--kadastara-ai-backend-predict.modal.run"

TILE = 512          # model input size
OVERLAP = 64        # pixels of overlap between tiles to hide seams
MAX_PREVIEW = 1500  # max dimension of the preview shown in the UI


def load_image(uploaded_file):
    """Read uploaded image, including GeoTIFFs."""
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


def tile_image(img, tile=TILE, overlap=OVERLAP):
    """Split image into overlapping tiles. Returns list of (PIL patch, x, y)."""
    w, h = img.size
    step = tile - overlap
    tiles = []
    for y in range(0, h, step):
        for x in range(0, w, step):
            box = (x, y, min(x + tile, w), min(y + tile, h))
            patch = img.crop(box)
            # Pad if patch is smaller than tile
            if patch.size != (tile, tile):
                padded = Image.new("RGB", (tile, tile), (0, 0, 0))
                padded.paste(patch, (0, 0))
                patch = padded
            tiles.append((patch, x, y, box))
    return tiles


def stitch_masks(tiles, mask_size, orig_size):
    """Stitch mask tiles back into the original image size."""
    mask = np.zeros((orig_size[1], orig_size[0]), dtype=np.uint8)
    weight = np.zeros_like(mask, dtype=np.float32)

    for patch, x, y, box in tiles:
        m = mask_size[id(patch)]
        x2, y2 = box[2], box[3]
        h, w = y2 - y, x2 - x
        mask[y:y2, x:x2] = np.maximum(mask[y:y2, x:x2], m[:h, :w])
        weight[y:y2, x:x2] += 1

    return mask


uploaded_file = st.file_uploader(
    "Choose a drone image",
    type=["tif", "tiff", "png", "jpg", "jpeg"]
)

model_choice = st.selectbox(
    "Select model to run",
    ["buildings", "roads", "water_bodies"]
)

threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.05)

if uploaded_file is not None:
    try:
        image = load_image(uploaded_file)
    except Exception as e:
        st.error(f"Could not read this image: {e}")
        st.stop()

    st.write(f"Image size: {image.size[0]} × {image.size[1]} pixels")

    # Show small preview only
    preview = image.copy()
    preview.thumbnail((MAX_PREVIEW, MAX_PREVIEW))
    st.image(preview, caption="Preview (downscaled for display)", width="stretch")

    if st.button("🚀 Run AI Analysis"):
        tiles = tile_image(image)
        st.write(f"Processing {len(tiles)} tiles of {TILE}×{TILE} pixels...")

        progress = st.progress(0)
        mask_size = {}

        for i, (patch, x, y, box) in enumerate(tiles):
            buf = io.BytesIO()
            patch.save(buf, format="PNG")
            files = {"file": ("tile.png", buf.getvalue(), "image/png")}
            params = {"model_name": model_choice, "threshold": threshold}

            try:
                r = requests.post(BACKEND_URL, files=files, params=params, timeout=300)
                r.raise_for_status()
                m = np.array(Image.open(io.BytesIO(r.content)).convert("L"))
                mask_size[id(patch)] = m
            except requests.exceptions.RequestException as e:
                st.error(f"Tile {i} failed: {e}")
                st.stop()

            progress.progress((i + 1) / len(tiles))

        full_mask = stitch_masks(tiles, mask_size, image.size)

        # Show downscaled result
        result_img = Image.fromarray(full_mask)
        result_preview = result_img.copy()
        result_preview.thumbnail((MAX_PREVIEW, MAX_PREVIEW))

        st.subheader(f"Result: {model_choice.capitalize()}")
        st.image(result_preview, caption=f"{model_choice} mask (preview)", width="stretch")

        # Download full-size mask
        out_buf = io.BytesIO()
        result_img.save(out_buf, format="PNG")
        st.download_button(
            "📥 Download full-resolution mask (PNG)",
            data=out_buf.getvalue(),
            file_name=f"{model_choice}_mask.png",
            mime="image/png",
        )
        st.success("Analysis complete!")
