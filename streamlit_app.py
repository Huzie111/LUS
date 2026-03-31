import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import gdown
import os

# Download model from Google Drive
@st.cache_resource
def load_model_from_drive():
    from model import LungUltrasoundModel
    
    # Your Google Drive file ID (from the shareable link)
    file_id = "1df5sydv-k8nmIh83mWK-vE8lxTfYQG9C"  # Replace with actual ID
    model_path = "pytorch_model.bin"
    
    if not os.path.exists(model_path):
        with st.spinner("Downloading model (44 MB)... This may take a minute"):
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, model_path, quiet=False)
    
    model = LungUltrasoundModel(num_classes=3, num_seg_classes=1)
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

# Load model
model = load_model_from_drive()
