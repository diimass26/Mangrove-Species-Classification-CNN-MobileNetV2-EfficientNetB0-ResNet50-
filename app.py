import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import time

# Load models
@st.cache_resource
def load_models():
    mobilenet_model = tf.keras.models.load_model('models/mobilenet_mangrove_best.h5')
    efficientnet_model = tf.keras.models.load_model('models/efficientnet_mangrove_best.h5')
    resnet_model = tf.keras.models.load_model('models/resnet_mangrove_best.h5')
    return mobilenet_model, efficientnet_model, resnet_model

mobilenet_model, efficientnet_model, resnet_model = load_models()

# Konstanta
IMAGE_SIZE = (224, 224)
CLASS_NAMES = ['Bruguiera Gymnorrhiza', 'Lumnitzera Littorea', 'Rhizophora Apiculata', 'Sonneratia Alba', 'Xylocarpus Granatum']  # Ganti sesuai datasetmu

st.title("🌿 Klasifikasi Spesies Mangrove dengan CNN")
st.write("Upload gambar daun mangrove untuk menguji model MobileNet, EfficientNet, dan ResNet.")

uploaded_file = st.file_uploader("Unggah gambar daun...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Load and resize image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar asli", use_column_width=True)

    image_resized = image.resize(IMAGE_SIZE)
    st.image(image_resized, caption="Gambar resize (224x224)", use_column_width=False)

    # Preprocess untuk prediksi
    img_array = np.array(image_resized) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)

    st.subheader("📊 Hasil Prediksi:")

    # Fungsi bantu untuk prediksi
    def predict_and_time(model, name):
        start = time.time()
        pred = model.predict(img_batch)
        duration = time.time() - start
        confidence = np.max(pred)
        predicted_class = CLASS_NAMES[np.argmax(pred)]
        return predicted_class, confidence, duration

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### MobileNetV2")
        pred, conf, dur = predict_and_time(mobilenet_model, "MobileNetV2")
        st.write(f"**Prediksi:** {pred}")
        st.write(f"**Confidence:** {conf:.2f}")
        st.write(f"⏱️ Waktu: {dur:.3f} detik")

    with col2:
        st.markdown("### EfficientNetB0")
        pred, conf, dur = predict_and_time(efficientnet_model, "EfficientNetB0")
        st.write(f"**Prediksi:** {pred}")
        st.write(f"**Confidence:** {conf:.2f}")
        st.write(f"⏱️ Waktu: {dur:.3f} detik")

    with col3:
        st.markdown("### ResNet50")
        pred, conf, dur = predict_and_time(resnet_model, "ResNet50")
        st.write(f"**Prediksi:** {pred}")
        st.write(f"**Confidence:** {conf:.2f}")
        st.write(f"⏱️ Waktu: {dur:.3f} detik")

    st.success("Prediksi selesai! Silakan analisis perbandingannya.")
