import streamlit as st
from PIL import Image
import numpy as np
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.utils import draw_bounding_boxes

weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
categories = weights.meta["categories"]
img_preprocess = weights.transforms()

@st.cache_resource
def load_model():
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.5)
    model.eval()
    return model

model = load_model()

def make_prediction(img_tensor):
    with torch.no_grad():
        prediction = model(img_tensor.unsqueeze(0))[0]
    mask = (prediction["labels"] == 1) | (prediction["labels"] == 9)
    prediction["boxes"] = prediction["boxes"][mask]
    prediction["labels"] = prediction["labels"][mask]
    prediction["scores"] = prediction["scores"][mask]
    prediction["labels"] = [categories[label] for label in prediction["labels"]]
    return prediction

def create_image_with_bboxes(img_np, prediction):
    img_tensor = torch.tensor(img_np)
    img_with_bboxes = draw_bounding_boxes(
        img_tensor, boxes=prediction["boxes"], labels=prediction["labels"],
        colors=["red" if label == "person" else "green" for label in prediction["labels"]],
        width=5
    )
    img_with_bboxes_np = img_with_bboxes.permute(1, 2, 0).numpy()  # (C, H, W) -> (H, W, C)
    return img_with_bboxes_np

# Streamlit Layout
st.title("Suchdrohne Machbarkeitsanalyse")

# Linke Spalte für Koordinaten und Auswahl
col1 = st.sidebar
latitude = col1.number_input("Breitengrad", format="%.6f")
longitude = col1.number_input("Längengrad", format="%.6f")

search_path = col1.radio("Suchpfad auswählen", ("Typ 1", "Typ 2", "Typ 3"))
if col1.button("Suche starten"):
    st.write("Suche läuft...")

# Rechte Spalte für Webcam und Buttons
col2 = st.container()
with col2:
    # Webcam Bildanzeige
    webcam_image = st.camera_input("Webcam")
    if webcam_image is not None:
        img = Image.open(webcam_image)
        img_np = np.array(img)
        img_tensor = img_preprocess(torch.tensor(img_np).permute(2, 0, 1) / 255.0)
        
        prediction = make_prediction(img_tensor)
        img_with_bbox = create_image_with_bboxes(img_tensor.mul(255).byte(), prediction)

        st.image(img_with_bbox, caption="Erkanntes Bild", use_column_width=True)

    # Layout für die großen Buttons unter der Webcam
    button_col1, button_col2 = st.columns([1, 1], gap="medium")

    # Große Buttons mit CSS
    button_style = """
    <style>
    .big-button {
        width: 100%;
        height: 60px;
        font-size: 20px;
    }
    </style>
    """
    st.markdown(button_style, unsafe_allow_html=True)

    # Not-Aus Button
    with button_col1:
        if st.button("Not-Aus", key="not_aus", use_container_width=True, help="Deaktiviert alle Operationen sofort", args=("big-button",)):
            st.write("Not-Aus aktiviert!")

    # Status Button
    with button_col2:
        if st.button("Status", key="status", use_container_width=True, help="Zeigt den aktuellen Systemstatus an", args=("big-button",)):
            st.write("Status abgerufen.")
