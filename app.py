import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image

# page config
st.set_page_config(page_title="Soybean Intel", layout="wide")

# custom css
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #2e7d32; color: white; font-weight: bold; }
    [data-testid="stMetricValue"] { font-size: 28px; color: #2e7d32; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_my_model():
    # load the trained model
    return tf.keras.models.load_model('soybean_weed_model.keras')

model = load_my_model()

# class names loaded from model
# 1. broadleaf, 2. grass, 3. soil, 4. soybean
CLASS_NAMES = ['Broadleaf (Weed)', 'Grass (Weed)', 'Soil', 'Soybean']

# adding a sidebar with some extra info and warnings
with st.sidebar:
    st.title("Project Info")
    st.markdown("---")
    st.info("This system uses **MobileNetV3-Large** to identify different types of weeds within soybean crops. The model is designed to analyze images of established soybean crops to detect weed growth. If you upload an image that is mostly soil, it will be labeled as soil, even if some weeds are present.")
    confidence_threshold = st.slider("Confidence Threshold Warning", 0.0, 1.0, 0.7)
    st.write("Current Model: `MobileNetV3Large`")

# main UI
st.title("Soybean Weed Detector")
st.caption("Agricultural Computer Vision Analysis")
st.write("---")

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    uploaded_file = st.file_uploader("Upload Field Image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Field Preview", use_container_width=True)

with col2:
    if uploaded_file is not None:
        # preprocessing
        # MobileNetV3Large expects 224x224 and 0-255 RGB integers
        img = image.resize((224, 224))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) 

        if st.button('Run Neural Analysis'):
            with st.spinner("Processing spectral features..."):
                # run the prediction 
                predictions = model.predict(img_array)
                
                # extract probabilities
                probabilities = predictions[0]
                
                class_index = np.argmax(probabilities)
                result = CLASS_NAMES[class_index]
                confidence = probabilities[class_index]

                # display result etrics
                m1, m2 = st.columns(2)
                m1.metric("Top Prediction", result)
                m2.metric("Confidence Score", f"{confidence:.2%}")

                # threshold warning
                if confidence < confidence_threshold:
                    st.warning(f"⚠️ Low Certainty: Prediction is below {confidence_threshold:.0%} threshold.")

                # probability visualization
                st.write("### Analysis Breakdown")
                chart_data = pd.DataFrame({
                    'Category': CLASS_NAMES,
                    'Probability': probabilities
                }).sort_values(by='Probability', ascending=True)
                
                st.bar_chart(chart_data, x='Category', y='Probability', horizontal=True, color="#2e7d32")

                # technical insights
                with st.expander("View Model Metadata"):
                    st.json({
                        "Architecture": "MobileNetV3-Large",
                        "Input_Shape": "224x224x3",
                        "Raw_Output_Array": probabilities.tolist(),
                        "Class_Map": {i: name for i, name in enumerate(CLASS_NAMES)}
                    })
    else:
        st.info("Upload an image of the field to begin the classification process.")