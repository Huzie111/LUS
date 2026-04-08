import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import gdown
import os



# Page config
st.set_page_config(page_title="Lung Ultrasound AI", page_icon="🫁", layout="wide")

st.title("🫁 Lung Ultrasound AI Assistant")
st.markdown("---")

CLASS_NAMES = ['COVID-19', 'Other Disease', 'Healthy']
device = 'cpu'

# Google Drive file ID - REPLACE WITH YOUR ACTUAL ID
FILE_ID = "1df5sydv-k8nmIh83mWK-vE8lxTfYQG9C"  # <--- CHANGE THIS
MODEL_PATH = "pytorch_model.bin"

@st.cache_resource
def load_model():
    from model import LungUltrasoundModel
    
    # Download model if not exists
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model (44 MB)... First time setup may take a minute"):
            url = f"https://drive.google.com/uc?id={FILE_ID}"
            gdown.download(url, MODEL_PATH, quiet=False)
            
    
    # Load model
    model = LungUltrasoundModel(num_classes=3, num_seg_classes=1)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

# Load model
try:
    model = load_model()
    
except Exception as e:
    st.error(f"Error loading model: {e}")
    model = None

def preprocess_image(image):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize((224, 224))
    img_array = np.array(image).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
    return img_tensor, np.array(image)

def create_overlay(original_img, seg_mask):
    h, w = original_img.shape[:2]
    mask_resized = cv2.resize(seg_mask.astype(np.float32), (w, h))
    overlay = original_img.copy()
    overlay[:, :, 0] = overlay[:, :, 0] + (mask_resized * 128).astype(np.uint8)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return overlay

def predict(image):
    img_tensor, original_img = preprocess_image(image)
    
    with torch.no_grad():
        logits, seg_logits = model(img_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_class].item()
        seg_probs = torch.sigmoid(seg_logits[0, 0]).cpu().numpy()
        seg_mask = (seg_probs > 0.5).astype(np.float32)
    
    return {
        'class': CLASS_NAMES[pred_class],
        'confidence': confidence,
        'probabilities': {
            'COVID-19': probs[0, 0].item(),
            'Other Disease': probs[0, 1].item(),
            'Healthy': probs[0, 2].item()
        },
        'seg_mask': seg_mask,
        'seg_probs': seg_probs,
        'original_img': original_img
    }

# Layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("📷 Upload Image")
    uploaded_file = st.file_uploader("Choose a lung ultrasound image...", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

with col2:
    st.subheader("📊 Results")
    
    if uploaded_file is not None and model is not None:
        with st.spinner("Analyzing..."):
            img_np = np.array(image)
            result = predict(img_np)
        
        st.markdown(f"### 🏥 Prediction: **{result['class']}**")
        st.markdown(f"### 📈 Confidence: **{result['confidence']:.1%}**")
        
        st.markdown("### 📊 Probability Breakdown")
        st.bar_chart(result['probabilities'])
        
        st.markdown("### 🎯 B-line Detection")
        overlay = create_overlay(result['original_img'], result['seg_mask'])
        st.image(overlay, caption="B-lines in red", use_container_width=True)
        
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(result['seg_probs'], cmap='hot', vmin=0, vmax=1)
        ax.set_title('B-line Probability')
        ax.axis('off')
        plt.colorbar(im, ax=ax)
        st.pyplot(fig)
        plt.close()

st.markdown("---")
st.markdown("⚠️ Research tool. Always consult a healthcare professional.")
