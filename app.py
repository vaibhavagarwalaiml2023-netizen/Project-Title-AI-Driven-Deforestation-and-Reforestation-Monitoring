import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
import tensorflow as tf
from PIL import Image

# --- Model aur Zaroori Cheezein Load Karna ---

# Apne behtar model ka naam yahaan likhein
MODEL_PATH = 'waste_sorter_model_v2.h5'
CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
IMG_HEIGHT = 180
IMG_WIDTH = 180

# MobileNetV2 ka preprocessing function
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

# Model ko load karna (cache karna taaki baar-baar load na ho)
@st.cache(allow_output_mutation=True)
def load_app_model():
    model = load_model(MODEL_PATH)
    print("Model loaded successfully!")
    return model

model = load_app_model()

# --- Helper Functions (Jaise humne notebook mein banaye the) ---

def get_sorting_instruction(predicted_class):
    """
    Yeh function model ke prediction ke aadhaar par
    user ko nirdesh (instruction) deta hai.
    """
    if predicted_class == 'paper':
        return "Nirdesh: Ise 'Paper Recycling' (Neela) bin mein daalein."
    elif predicted_class == 'cardboard':
        return "Nirdesh: Ise 'Cardboard Recycling' (Neela) bin mein daalein."
    elif predicted_class == 'plastic':
        return "Nirdesh: Ise 'Plastic Recycling' (Neela) bin mein daalein."
    elif predicted_class == 'glass':
        return "Nirdesh: Ise 'Glass Recycling' (Peela) bin mein daalein."
    elif predicted_class == 'metal':
        return "Nirdesh: Ise 'Metal Recycling' (Laal) bin mein daalein."
    elif predicted_class == 'trash':
        return "Nirdesh: Yeh recycle nahi ho sakta. Ise 'General Waste' (Kaala) bin mein daalein."
    else:
        return "Nirdesh: Is item ko pehchaan nahi pa raha hoon."

def model_predict(image, model):
    """
    Image ko preprocess karke prediction karta hai
    """
    # Image ko zaroori size mein badalna
    img = image.resize((IMG_HEIGHT, IMG_WIDTH))
    img_array = img_to_array(img)
    
    # Naye model ke liye preprocess karna
    img_array_processed = preprocess_input(img_array) 
    
    img_array_expanded = tf.expand_dims(img_array_processed, 0) # Create a batch

    # Prediction karna
    predictions = model.predict(img_array_expanded)
    return predictions

# --- Streamlit Web App ka UI ---

st.title("AI Waste Sorter ♻️")
st.header("Kooda (Waste) Pehchaane aur Sahi Nirdesh Paayein")
st.write("Apne kooda item ki ek image upload karein, aur AI batayega ki use kaise sort karna hai.")

# Image Upload button
uploaded_file = st.file_uploader("Yahaan ek image chunein...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Uploaded image ko dikhana
    image = Image.open(uploaded_file)
    st.image(image, caption='Aapki Upload ki hui Image.', use_column_width=True)
    st.write("")
    
    # Prediction karne ke liye "loading" state dikhana
    with st.spinner('AI soch raha hai...'):
        # Prediction karna
        predictions = model_predict(image, model)
        score = tf.nn.softmax(predictions[0])

        # Result dikhana
        predicted_class = CLASS_NAMES[np.argmax(score)]
        confidence = 100 * np.max(score)
        instruction = get_sorting_instruction(predicted_class)

    st.success("Prediction Poori Hui!")
    
    st.subheader(f"Pehchaan (Identify): Yeh {predicted_class} hai.")
    st.write(f"Confidence (Kitna pakka hai): {confidence:.2f}%")
    
    st.subheader(instruction)