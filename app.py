import streamlit as st
import requests
import json

# --- Beam Configuration ---
BEAM_ENDPOINT = "https://kadastara-backend-abc123.app.beam.cloud"  # Replace with your URL
BEAM_TOKEN = "YOUR_BEAM_API_TOKEN"  # Replace with your token

st.title("🗺️ Kadastara – AI Cadastral Mapping Platform")

model_type = st.selectbox("Feature Type", ["Buildings", "Roads", "Water Bodies"])
threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5)

uploaded_file = st.file_uploader("Upload Drone Image", type=["tif", "tiff", "png", "jpg"])

if uploaded_file and st.button("🚀 Run AI Analysis"):
    with st.spinner("Processing on Beam GPU..."):
        files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
        data = {"model": model_type, "threshold": threshold}
        headers = {"Authorization": f"Bearer {BEAM_TOKEN}"}

        resp = requests.post(
            BEAM_ENDPOINT,
            files=files,
            data=data,
            headers=headers,
            timeout=600,
        )

        if resp.status_code == 200:
            result = resp.json()
            st.success("Analysis complete!")
            st.json(result["report"])
            st.download_button(
                "📥 Download GeoJSON",
                data=json.dumps(result["geojson"]),
                file_name="parcels.geojson",
                mime="application/json",
            )
        else:
            st.error(f"Backend error: {resp.status_code} — {resp.text}")
