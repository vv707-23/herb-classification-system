import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers, models
from PIL import Image
import numpy as np

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224,224,3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(15, activation='softmax')
])

model.build((None,224,224,3))
model.load_weights("model.weights.h5")


# Class names (correct order)
class_names = ['Aloevera', 'Arali', 'Beans', 'Citron lime (herelikai)', 'Curry',
               'Eucalyptus', 'Honge', 'Kambajala', 'Malabar_Nut', 'Neem',
               'Nooni', 'Pea', 'Pepper', 'Sapota', 'Taro']

st.title("🌿 Ayurvedic Herb Identification System")

uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocessing
    img = image.resize((224, 224))
    img = np.array(img)

    if img.shape[-1] == 4:
        img = img[:, :, :3]

    img_array = img / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)[0]

    # Normalize (quick fix)
    prediction = prediction / prediction.sum()

    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]
    confidence = prediction[predicted_index]

    st.success(f"🌿 Predicted Herb: {predicted_class}")
    st.write(f"Confidence: {confidence*100:.2f}%")