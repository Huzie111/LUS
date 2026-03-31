import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import os

# Page configuration
st.set_page_config(
    page_title="Lung Ultrasound AI",
    page_icon="🫁",
    layout="wide"
)

# Title
st.title("🫁 Lung Ultrasound AI Assistant")
st.markdown("---")

# Class names
CLASS_NAMES = ['COVID-19', 'Other Disease', 'Healthy']
device = torch.device('cpu')

# Load model
@st.cache_resource
def load_model():
    """Load the model once and cache it"""
    from model import LungUltrasoundModel
    model = LungUltrasoundModel(num_classes=3, num_seg_classes=1)
    model_path = "pytorch_model.bin"
    
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        st.success("✅ Model loaded successfully!")
        return model
    else:
        st.error(f"❌ Model file not found: {model_path}")
        return None

# Load model
model = load_model()

def preprocess_image(image):
    """Preprocess uploaded image for model input"""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize((224, 224))
    img_array = np.array(image).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
    return img_tensor, np.array(image.resize((224, 224)))

def create_overlay(original_img, seg_mask):
    """Create red overlay for segmentation"""
    h, w = original_img.shape[:2]
    mask_resized = cv2.resize(seg_mask.astype(np.float32), (w, h))
    overlay = original_img.copy()
    overlay[:, :, 0] = overlay[:, :, 0] + (mask_resized * 128).astype(np.uint8)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return overlay

def predict(image):
    """Run inference on uploaded image"""
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

# Create two columns for layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("📷 Upload Ultrasound Image")
    uploaded_file = st.file_uploader(
        "Choose a lung ultrasound image...",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a lung ultrasound image in JPG or PNG format"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

with col2:
    st.subheader("📊 Results")
    
    if uploaded_file is not None and model is not None:
        with st.spinner("Analyzing image..."):
            # Convert PIL to numpy
            img_np = np.array(image)
            result = predict(img_np)
        
        # Display prediction
        st.markdown(f"### 🏥 Prediction: **{result['class']}**")
        st.markdown(f"### 📈 Confidence: **{result['confidence']:.1%}**")
        
        # Display probabilities as a bar chart
        st.markdown("### 📊 Probability Breakdown")
        prob_data = {
            'COVID-19': result['probabilities']['COVID-19'],
            'Other Disease': result['probabilities']['Other Disease'],
            'Healthy': result['probabilities']['Healthy']
        }
        st.bar_chart(prob_data)
        
        # Display segmentation overlay
        st.markdown("### 🎯 B-line Detection")
        
        overlay = create_overlay(result['original_img'], result['seg_mask'])
        st.image(overlay, caption="Segmentation Overlay (B-lines in red)", use_container_width=True)
        
        # Display heatmap
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(result['seg_probs'], cmap='hot', vmin=0, vmax=1)
        ax.set_title('B-line Probability Heatmap')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        st.pyplot(fig)
        plt.close()

elif uploaded_file is None:
    st.info("👈 Upload a lung ultrasound image to get started")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    <p>⚠️ This is a research tool. Always consult a healthcare professional for diagnosis.</p>
    <p>Model: EfficientNet-B3 + SegFormer | Trained on 1,463 images from Ugandan hospitals</p>
    </div>
    """,
    unsafe_allow_html=True
)