import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt") 

model = load_model()

st.title("Classificador de Imagens com YOLOv8 + COCO")

option = st.radio("Escolha o modo de entrada:", ('Imagem local', 'Webcam'))

def classify_and_display(image: Image.Image):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        image.save(tmp.name)
        results = model(tmp.name)

    # Exibe imagem com bounding boxes
    results[0].show()

    # Renderiza imagem com caixas
    result_img = results[0].plot()
    st.image(result_img, caption="Resultado da Detecção", use_container_width=True)

    # Mostra classes detectadas
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        st.write(f"Classe: **{model.names[cls_id]}** | Confiança: **{conf:.2f}**")

# 
if option == 'Imagem local':
    uploaded_file = st.file_uploader("Escolha uma imagem...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Imagem carregada", use_container_width=True)

        if st.button("Classificar"):
            with st.spinner("Processando..."):
                classify_and_display(image)

else:
    picture = st.camera_input("Capturar imagem da webcam")

    if picture is not None:
        image = Image.open(picture).convert("RGB")
        st.image(image, caption="Imagem capturada", use_container_width=True)

        if st.button("Classificar"):
            with st.spinner("Processando..."):
                classify_and_display(image)
