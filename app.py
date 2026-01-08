import streamlit as st
uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])

from PIL import Image
image = Image.open(uploaded_file)
img_gray = image.convert('L')
img_resized = img_gray.resize((8, 8))

import numpy as np
img_array = np.array(img_resized)
img_array = img_array / 16.0
img_flat = img_array.flatten().reshape(1, -1)

from sklearn.neural_network import MLPClassifier
prediction = model.predict(img_flat)[0]
probs = model.predict_proba(img_flat)[0]
