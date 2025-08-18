import streamlit as st
import pickle
import numpy as np
import os

st.set_page_config(
    page_title="Petalytics-Iris Flower Classifier",
    page_icon="🌸",
    layout="centered"
)

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #e0eafc 0%, #cfdef3 100%);
    }
    .prediction-row {
        display: flex;
        align-items: center;
        justify-content: center;
        flex-direction: row;
        margin-top: 1em;
        gap: 1.2em;
    }
    .prediction-name {
        font-size: 1.25em;
        font-weight: bold;
        background: #e0eafc;
        padding: 0.25em 1.3em;
        border-radius: 1em;
        color: #17437c;
        text-shadow: 0 2px 6px #f2f6fa;
    }
    .center-image {
        display: flex;
        justify-content: center;
        margin-top: 1em;
        margin-bottom: 1em;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🌸Petalytics-Iris Flower Classifier")
st.markdown("Predict Iris species from flower measurements with a beautiful centered preview.")

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.subheader("Enter Flower Measurements")
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.1, step=0.1)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.5, step=0.1)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=1.4, step=0.1)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=0.2, step=0.1)
user_values = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

flower_emojis = {
    "Iris-setosa": "🌸",
    "Iris-versicolor": "🌻",
    "Iris-virginica": "🌺"
}
flower_images = {
    "Iris-setosa": "images/setosa.jpg",
    "Iris-versicolor": "images/versicolor.jpg",
    "Iris-virginica": "images/virginica.jpg",
}

col1, col2 = st.columns([1, 2])
with col1:
    predict_clicked = st.button("🌼 Predict Flower Type")
with col2:
    # Show prediction beside button when clicked
    if predict_clicked:
        prediction = model.predict(user_values)[0]
        emoji = flower_emojis.get(prediction, "")
        name_display = f"{emoji} <span class='prediction-name'>{prediction}</span> {emoji}"
        st.markdown(name_display, unsafe_allow_html=True)
        st.balloons()

# Centered image below prediction row
if 'predict_clicked' in locals() and predict_clicked:
    image_path = None
    if prediction == "Iris-setosa":
        image_path = flower_images["Iris-setosa"]
    elif prediction == "Iris-versicolor":
        image_path = flower_images["Iris-versicolor"]
    elif prediction == "Iris-virginica":
        image_path = flower_images["Iris-virginica"]
    if image_path and os.path.exists(image_path):
        st.markdown(
            "<div class='center-image'>", unsafe_allow_html=True
        )
        st.image(image_path, width=340)
        st.markdown(
            "</div>", unsafe_allow_html=True
        )

st.markdown("""
---
<small><i>Developed with Streamlit · Beautifully centered image & label · By [Your Name]</i></small>
""", unsafe_allow_html=True)
