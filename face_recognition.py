import streamlit as st
from PIL import Image
import io
from autogluon.vision import ImagePredictor
import pandas as pd
import requests


def load_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes))
    return img


def evaluate_image(image):
    # model = joblib.load(model)
    # model.predict(image)
    return  # result


url = 'https://upcdn.io/FW25c67/raw/image_predictor'
response = requests.get(url)
# predictor = ImagePredictor.load('D:\\suml\\image_predictor')
with open('image_predictor', 'wb') as f:
    f.write(response.content)
predictor = ImagePredictor.load('image_predictor')

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

photo = st.file_uploader("Upload photo", type=["jpg", "jpeg", "png"])
if photo:
    bytes = photo.getvalue()
    image = load_image(photo.read())

    with open("temp_image.jpg", "wb") as f:
        f.write(photo.getbuffer())

    test_df = pd.DataFrame(["temp_image.jpg"], columns=['image'])

    predictions = predictor.predict(test_df)
    for  pred in zip( predictions):
        st.write(pred)
    resized_image = image.resize((250, 250))
    st.image(resized_image)
    st.write("")

evaluate_photo = st.button("Evaluate!", use_container_width=True)

if evaluate_photo:
    result = evaluate_image(image)
    st.success(result)
