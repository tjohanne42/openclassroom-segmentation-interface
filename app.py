import os
import io
import streamlit as st
from PIL import Image
import requests
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

NUM_CLASSES = 8
API_URL = "https://my-flask-api-917968940784.europe-west1.run.app/predict"
IMAGE_DIR = "test_images"
MASK_DIR = "test_masks"
ALPHA = 0.5

st.set_page_config(page_title="Segmentation d'Image", layout="wide")
st.title("🖼️ Segmentation d'Image avec IA")
st.markdown("Testez votre modèle de segmentation et comparez facilement les résultats.")

@st.cache_data
def load_image(path):
    return Image.open(path).convert("RGB")

@st.cache_data
def load_mask(path):
    return Image.open(path).convert("L")

@st.cache_data
def get_image_files(directory):
    return sorted([f for f in os.listdir(directory) if f.lower().endswith((".png", ".jpg", ".jpeg"))])

cmap = matplotlib.colors.ListedColormap(
    matplotlib.colormaps["jet"](np.linspace(0, 1, NUM_CLASSES))[:, :3]
)

def normalize_mask(mask_np, num_classes):
    if mask_np.max() > num_classes - 1:
        mask_np = (mask_np.astype(np.float32) / 255 * (num_classes - 1)).round().astype(np.uint8)
    return mask_np

def generate_colored_mask(mask_np, cmap, num_classes):
    mask_np = normalize_mask(mask_np, num_classes)
    palette = (cmap(np.arange(num_classes))[:, :3] * 255).astype(np.uint8)
    colored_mask = np.zeros((*mask_np.shape, 3), dtype=np.uint8)
    for i in range(num_classes):
        colored_mask[mask_np == i] = palette[i]
    return Image.fromarray(colored_mask)

def overlay_mask(image_np, mask_color_np):
    return Image.fromarray((ALPHA * image_np + (1 - ALPHA) * mask_color_np).astype(np.uint8))

image_files = get_image_files(IMAGE_DIR)
if not image_files:
    st.sidebar.error("Aucune image trouvée dans le dossier d'images.")
    st.stop()

st.sidebar.title("🔍 Options")
selected_file = st.sidebar.selectbox("Choisissez une image de test", image_files)
image_path = os.path.join(IMAGE_DIR, selected_file)
mask_file = selected_file.replace("image", "mask")
mask_path = os.path.join(MASK_DIR, mask_file)

original_image = load_image(image_path)
original_np = np.array(original_image)

gt_mask_color, overlay_gt = None, None
if os.path.exists(mask_path):
    gt_mask = load_mask(mask_path)
    gt_mask_np = np.array(gt_mask)
    gt_mask_color = generate_colored_mask(gt_mask_np, cmap, NUM_CLASSES).resize(original_image.size)
    overlay_gt = overlay_mask(original_np, np.array(gt_mask_color))

st.subheader("📌 Données de Vérité Terrain")
col1, col2, col3 = st.columns(3)
with col1:
    st.image(original_image, caption="Image Originale", use_container_width=True)
with col2:
    st.image(gt_mask_color if gt_mask_color else np.zeros_like(original_np), caption="Masque de Vérité Terrain", use_container_width=True)
with col3:
    st.image(overlay_gt if overlay_gt else np.zeros_like(original_np), caption="Overlay Vérité Terrain", use_container_width=True)

st.sidebar.subheader("⚡ Prédiction")
if st.sidebar.button("Analyser l'image 🚀"):
    with st.spinner("⏳ Analyse en cours..."):
        with open(image_path, "rb") as f:
            response = requests.post(API_URL, files={"file": f})
    if response.status_code == 200:
        st.sidebar.success("✅ Prédiction terminée !")
        pred_mask = Image.open(io.BytesIO(response.content)).convert("L")
        pred_mask_np = np.array(pred_mask)
        pred_mask_color = generate_colored_mask(pred_mask_np, cmap, NUM_CLASSES).resize(original_image.size)
        overlay_pred = overlay_mask(original_np, np.array(pred_mask_color))
        
        st.subheader("📊 Résultats de la Prédiction")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(overlay_gt if overlay_gt else original_image, caption="Overlay Vérité Terrain", use_container_width=True)
        with col2:
            st.image(overlay_pred, caption="Overlay Prédiction", use_container_width=True)
        with col3:
            st.image(pred_mask_color, caption="Masque Prédit", use_container_width=True)
    else:
        st.sidebar.error(f"❌ Erreur lors de la prédiction : {response.status_code}")


if __name__ == "__main__":
    os.system("streamlit run app.py --server.port=8080 --server.address=0.0.0.0")