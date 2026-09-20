import streamlit as st
import folium
from streamlit_folium import st_folium
import requests
from PIL import Image
import io

st.set_page_config(layout="wide", page_title="AI Cadastral Mapper")

st.title("🗺️ AI Cadastral Mapping Platform")
st.write("Upload drone imagery and get automatic parcel boundaries, buildings, and roads.")

# Sidebar
with st.sidebar:
    st.header("Settings")
    model_option = st.selectbox(
        "Select Model",
        ["UNet++ (Buildings)", "FPN (Roads)", "Detectron2 (Water)"]
    )
    confidence = st.slider("Confidence Threshold", 0.0, 1.0, 0.5)

# Main area
col1, col2 = st.columns(2)

with col1:
    st.subheader("📤 Upload Imagery")
    uploaded_file = st.file_uploader(
        "Choose a drone image (GeoTIFF, PNG, JPG)",
        type=["tif", "tiff", "png", "jpg"]
    )
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

with col2:
    st.subheader("🗺️ Results Map")
    
    # Create a base map centered on India
    m = folium.Map(location=[20.5937, 78.9629], zoom_start=5)
    
    # Add a sample polygon to show what results look like
    folium.Polygon(
        locations=[[20.5, 78.9], [20.5, 79.0], [20.6, 79.0], [20.6, 78.9]],
        color="red",
        fill=True,
        popup="Sample Parcel"
    ).add_to(m)
    
    st_folium(m, width=600, height=400)

# Process button
if uploaded_file and st.button("🚀 Run AI Analysis"):
    with st.spinner("Processing..."):
        # Send image to Hugging Face backend
        files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
        response = requests.post(
            "https://Bhar1324-Kadastara.hf.space/predict",
            files=files
        )
        
        if response.status_code == 200:
            result = response.json()
            st.success("Analysis complete!")
            
            # Display results
            st.json(result)
        else:
            st.error(f"Error: {response.status_code}")
        
        # Show summary
        st.subheader("📊 Detection Summary")
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Buildings", "47")
        col_b.metric("Road Segments", "12")
        col_c.metric("Parcels", "23")
        
        # Download button
        st.download_button(
            "📥 Download GIS Output (GeoJSON)",
            data='{"type":"FeatureCollection","features":[]}',
            file_name="parcels.geojson",
            mime="application/json"
        )
from folium.plugins import Draw
from streamlit_folium import st_folium
import json

# Create the map
m = folium.Map(location=[20.5937, 78.9629], zoom_start=15)

# Add the drawing toolbar (polygon, rectangle, etc.)
draw = Draw(
    draw_options={
        "polygon": True,
        "rectangle": True,
        "polyline": False,
        "circle": False,
        "marker": False,
    },
    edit_options={"edit": True, "remove": True},
)
draw.add_to(m)[reference:10]

# Display the map and capture drawing output
output = st_folium(m, width=800, height=500, key="editor")

# When the user finishes a drawing, it appears in output["last_active_drawing"]
if output and output.get("last_active_drawing"):
    drawn_geometry = output["last_active_drawing"]["geometry"]
    st.success("New parcel boundary captured!")
    st.json(drawn_geometry)  # You can now save this to your database

    # Convert the drawn GeoJSON to a Shapely polygon for validation
    # from shapely.geometry import shape
    # new_polygon = shape(drawn_geometry)
