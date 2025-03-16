import os
import io
import streamlit as st
from PIL import Image
import requests
import numpy as np
# import matplotlib.cm as cm
import matplotlib

# Paramètres
NUM_CLASSES = 8
# API_URL = "http://localhost:5000/predict"  # local
API_URL = "https://my-flask-api-917968940784.europe-west1.run.app/predict" # deployed
IMAGE_DIR = "test_images"
MASK_DIR = "test_masks"

# Récupère la liste des images de test
image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

st.title("Interface de test pour la segmentation d'image")

if image_files:
    # Sélection d'une image
    selected_file = st.selectbox("Sélectionnez une image de test", image_files)
    image_path = os.path.join(IMAGE_DIR, selected_file)
    mask_path = os.path.join(MASK_DIR, selected_file.replace("image", "mask"))  # on suppose que les noms correspondent

    # Affichage de l'image originale
    st.subheader("Image originale")
    original_image = Image.open(image_path).convert("RGB")
    st.image(original_image, caption="Image de test", use_container_width=True)

    # Affichage du masque de vérité terrain
    st.subheader("Masque de vérité terrain")
    if os.path.exists(mask_path):
        # Charge le masque et convertit en niveaux de gris
        gt_mask = Image.open(mask_path).convert("L")
        gt_mask_np = np.array(gt_mask)
        # Les valeurs de masque sont des classes (0 à NUM_CLASSES-1), on les met à l'échelle pour les visualiser
        gt_mask_scaled = (gt_mask_np.astype(np.float32) * (255.0 / (NUM_CLASSES - 1))).astype(np.uint8)
        # Application d'une colormap pour améliorer la visualisation
        norm_gt_mask = gt_mask_scaled.astype(np.float32) / 255.0
        # cmap = cm.get_cmap('jet')
        cmap = matplotlib.colormaps.get_cmap('jet')
        gt_mask_color = (cmap(norm_gt_mask)[:, :, :3] * 255).astype(np.uint8)
        gt_mask_color = Image.fromarray(gt_mask_color)
        st.image(gt_mask_color, caption="Masque de vérité terrain (colorisé)", use_container_width=True)
    else:
        st.warning("Aucun masque correspondant trouvé.")

    # Lancement de la prédiction via l'API
    if st.button("Lancer la prédiction"):
        with open(image_path, "rb") as f:
            files = {"file": f}
            response = requests.post(API_URL, files=files)

        if response.status_code == 200:
            st.success("Prédiction réussie !")
            # L'API retourne uniquement le masque prédit au format PNG
            pred_mask = Image.open(io.BytesIO(response.content)).convert("L")
            pred_mask_np = np.array(pred_mask)
            # Application d'une colormap pour le masque prédit
            norm_pred_mask = pred_mask_np.astype(np.float32) / 255.0
            # cmap = cm.get_cmap('jet')
            cmap = matplotlib.colormaps.get_cmap('jet')
            pred_mask_color = (cmap(norm_pred_mask)[:, :, :3] * 255).astype(np.uint8)
            pred_mask_color = Image.fromarray(pred_mask_color)

            st.subheader("Masque prédit")
            st.image(pred_mask_color, caption="Masque prédit (colorisé)", use_container_width=True)

            # Création de l'overlay (fusion) entre l'image originale et le masque prédit colorisé
            original_np = np.array(original_image)
            pred_mask_color_np = np.array(pred_mask_color)
            alpha = 0.5  # coefficient de transparence
            overlay_np = (alpha * original_np + (1 - alpha) * pred_mask_color_np).astype(np.uint8)
            overlay_image = Image.fromarray(overlay_np)

            st.subheader("Overlay : Image originale + Masque prédit")
            st.image(overlay_image, caption="Overlay", use_container_width=True)
        else:
            st.error(f"Erreur lors de l'appel à l'API : {response.status_code}")
else:
    st.error("Aucune image trouvée dans le répertoire de test.")
