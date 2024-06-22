import streamlit as st
from PIL import Image
import io
import pandas as pd
import cv2
import tensorflow as tf
import numpy as np
import sys

sys.stdout.reconfigure(encoding="utf-8")


def load_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes))
    return img


def evaluate_image(image):
    # model = joblib.load(model)
    # model.predict(image)
    return  # result


st.set_page_config(page_title="Face recognition app")


def write_text_in_center(text, header):
    st.markdown(
        f"<{header} style='text-align: center;'>{text}</{header}>",
        unsafe_allow_html=True,
    )


write_text_in_center("Hello!", "h1")
write_text_in_center(
    "This is a simple program for evaluating face emotions based on photo!", "h2"
)
write_text_in_center("Please import your file below.", "h3")

model = tf.keras.models.load_model("face_recognition.keras")

class_names = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

photo = st.file_uploader("Upload photo", type=["jpg", "jpeg", "png"])
write_text_in_center("Or", "h3")
photo = st.camera_input("Take a picture")
if photo:
    image = Image.open(photo)
    # st.write(image.size)
    image_np = np.array(image)
    input_image = (
        image_np.reshape((1, 48, 48, 1)).astype("float32") / 255.0
    )  # Reshape and normalize
    # st.write(image_np.shape)
    # gray_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    # st.write(input_image.shape)
    result = model.predict(input_image)
    pred_labels = np.argmax(result, axis=1)

    st.write(class_names[int(pred_labels[0])])
    # st.write(result)

    # image_np = image_np / 255
    # image = cv2.cvtColor(image_np, cv2.IMREAD_GRAYSCALE)
    # image = cv2.resize(image, (50, 50))
    # result = model.predict(image)
    # st.write(result)
    bytes = photo.getvalue()

evaluate_photo = st.button("Evaluate!", use_container_width=True)

if evaluate_photo:
    result = evaluate_image(image)
    st.success(result)
