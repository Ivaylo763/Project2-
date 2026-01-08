import streamlit as st
import numpy as np
from PIL import Image
import joblib
import requests
import io

st.set_page_config(page_title = "Handwritten Digir Recongnition", page_icon="✍️")
st.title("✍️ Handwritten Digit Recongnition")
st.write("Upload a handwritten digit image and AI will try try to recohnize it.")


@st.cache_resource
def load_model():
  try:
    from sklearn.detasets import load_digits
    from sklearn.network import MLPClassifier
    from sklearn.model_seslection import train_test_split



digits = load_digits()
X = digits.images.reshape((len(digits.images), -1)) / 16.0
y = digits.targer
X_train, _, y_train,_ = train_test_split(X, y, test_size = 0.2, random_state = 42)

model MLPClassifier(
hidden_layer_sizes=(100,),
max_iter=100,
random state=42
)
model.fit(X_train, y_train)
return model
except Exception as e:
st.error(f"Model loading error: {e}")
return None

model load_model()

if model is None:
st.warning("Could not load model. Using fallback recognition.")
else:
st.success("Model loaded successfully!")


uploaded_file st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
# Display the uploaded image
image Image.open(uploaded_file)
st.image(image, caption="Uploaded Image', use_column_width=True)
