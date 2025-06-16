import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# === Load your trained model ===
model = load_model(
    r"C:\Users\Acer\Downloads\Python\Deep Learning\CNN\Brain Tumor\brain_tumor_classifier.keras"
)

# === Define categories ===
categories = ["glioma", "meningioma", "notumor", "pituitary"]

category_descriptions = {
    "pituitary": "A pituitary tumor is an abnormal growth in the pituitary gland, located at the base of the brain.",
    "meningioma": "A meningioma arises from the meninges, protecting the brain and spinal cord.",
    "glioma": "Gliomas originate in the glial cells of the brain or spinal cord.",
    "notumor": "No Tumor."
}

st.set_page_config(page_title="Brain Tumor Classifier")
st.title("🧠 Brain Tumor Detection")

# === File uploader ===
uploaded_file = st.file_uploader(
    "Upload an MRI Image",
    type=['jpg', 'jpeg', 'png'],
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded MRI Image', use_container_width=True)

    img = image.resize((128, 128))
    img_array = np.array(img) / 255.0

    if img_array.ndim == 2:
        img_array = np.stack((img_array,) * 3, axis=-1)
    elif img_array.shape[-1] == 1:
        img_array = np.concatenate([img_array] * 3, axis=-1)

    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class_raw = categories[np.argmax(prediction)]
    confidence = np.max(prediction)

    # Format prediction: capitalize properly
    if predicted_class_raw == "notumor":
        predicted_label = "No Tumor"
    else:
        predicted_label = predicted_class_raw.title()

    st.subheader(f"Prediction: **{predicted_label}**")
    st.write(f"Confidence: {confidence:.2f}")

    if predicted_class_raw != "notumor":
        st.info(f"**Description:** {category_descriptions[predicted_class_raw]}")
