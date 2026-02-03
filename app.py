import streamlit as st
import cv2
import numpy as np
import math
from PIL import Image
import os
import torch
import time

# PyTorch 2.6+ & Ultralytics Compatibility Patch (ULTRA-ROBUST)
# (torch is already imported above)

# 1. Broadly allow all ultralytics and common torch components in unpickling
try:
    import ultralytics.nn.tasks as tasks
    import ultralytics.nn.modules.conv as conv
    import ultralytics.nn.modules.block as block
    import ultralytics.nn.modules.head as head
    import torch.nn as nn
    
    if hasattr(torch.serialization, 'add_safe_globals'):
        torch.serialization.add_safe_globals([
            tasks.SegmentationModel, tasks.DetectionModel,
            conv.Conv, conv.Concat,
            block.C2f, block.DFL, block.Bottleneck,
            head.Segment, head.Detect,
            nn.modules.container.Sequential, nn.modules.container.ModuleList,
            nn.modules.conv.Conv2d, nn.modules.batchnorm.BatchNorm2d,
            nn.modules.activation.SiLU, nn.modules.upsampling.Upsample,
            nn.modules.pooling.MaxPool2d,
        ])
except Exception:
    pass

# 2. Forcibly disable weights_only for local model loading
original_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)
torch.load = patched_torch_load

from ultralytics import YOLO

# Page configuration
st.set_page_config(
    page_title="AI Palm Reader",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Light Theme / White Background
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
        color: #212529;
    }
    .stApp {
        background-color: #ffffff;
    }
    .report-card {
        background: #ffffff;
        padding: 15px;
        border-radius: 12px;
        border-left: 6px solid #007bff;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        color: #212529;
    }
    .line-title {
        color: #007bff;
        font-weight: bold;
        font-size: 1.1rem;
        text-transform: uppercase;
    }
    .interpretation {
        color: #495057;
        font-style: italic;
        font-size: 0.95rem;
    }
    h1, h2, h3 {
        color: #212529 !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# Cache the model
@st.cache_resource
def load_model():
    model_path = os.path.join("Models", "last.pt")
    if not os.path.exists(model_path):
        st.error(f"Model not found at {model_path}.")
        return None
    try:
        return YOLO(model_path)
    except Exception as e:
        if "UnpicklingError" in str(e) or "weights_only" in str(e):
            st.warning("Retrying model load with PyTorch compatibility patch...")
            return YOLO(model_path)
        else:
            st.error(f"Error loading model: {e}")
            return None

model = load_model()
class_map = {0: "head", 1: "heart", 2: "life"}

# Analysis Helper Functions
def analyze_line(points, line_type):
    if len(points) < 2: return "Analysis pending..."
    
    # Calculate Length
    length = np.sum(np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1)))
    
    # Calculate Curvature
    vecs = np.diff(points, axis=0)
    angles = []
    for i in range(1, len(vecs)):
        v1, v2 = vecs[i - 1], vecs[i]
        norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
        if norm_product > 1e-8:
            angles.append(np.arccos(np.clip(np.dot(v1, v2) / norm_product, -1.0, 1.0)))
    angle_deg = math.degrees(np.mean(angles)) if angles else 0

    if line_type == "heart":
        desc = "Emotionally expressive and intuitive." if angle_deg > 15 else "Practical and logical approach to feelings."
        desc += " High empathy." if length > 250 else " Private nature." if length < 150 else ""
    elif line_type == "head":
        if length > 250 and angle_deg < 10: desc = "Strong, logical mind; very organized."
        elif angle_deg > 20: desc = "Highly creative and imaginative mind."
        elif length < 150: desc = "Quick, impulsive thinker; focused on short-term goals."
        else: desc = "Balanced and practical thinker."
    elif line_type == "life":
        desc = "Strong vitality and resilience." if length > 300 else "Independent and adventurous." if length < 150 else "Balanced energy levels."
        desc += " Enthusiastic nature." if angle_deg > 30 else ""
    else: desc = "General palm feature detected."
    return desc

# Main App Layout
st.title("PALM READER")
st.markdown("A professional real-time hand detection and analysis system.")

# Sidebar Settings
st.sidebar.title("Settings")
confidence_threshold = st.sidebar.slider("Confidence", 0.1, 1.0, 0.5, 0.05)
input_mode = st.radio("Select Analysis Mode:", ["Live Hand Detection (Stream)", "Upload Image"], horizontal=True)

if input_mode == "Live Hand Detection (Stream)":
    st.info("Show your hand to the camera for real-time analysis.")
    run_stream = st.toggle("Start Live Detection", value=False)
    
    if run_stream:
        cap = cv2.VideoCapture(0)
        frame_placeholder = st.empty()
        report_placeholder = st.empty()
        
        while run_stream:
            ret, frame = cap.read()
            if not ret:
                st.error("Could not access camera.")
                break
            
            # Predict
            results = model.predict(source=frame, conf=confidence_threshold, task='segment', save=False, verbose=False)
            result = results[0]
            
            # Annotate
            annotated_frame = result.plot()
            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            
            frame_placeholder.image(annotated_frame_rgb, caption="Live Detection View", use_column_width=True)
            
            # Interpretation logic for live data
            interpretations = []
            if result.masks is not None and result.boxes is not None:
                masks = result.masks.xy
                class_ids = result.boxes.cls.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                
                best_indices = {}
                for i, cid in enumerate(class_ids):
                    if cid not in best_indices or confs[i] > confs[best_indices[cid]]:
                        best_indices[cid] = i
                
                for cid, idx in best_indices.items():
                    line_name = class_map.get(int(cid), "Unknown")
                    interp = analyze_line(masks[idx], line_name)
                    interpretations.append((line_name, interp))

            # Display reports dynamically
            if interpretations:
                report_html = '<div style="display: flex; flex-wrap: wrap; gap: 10px;">'
                for name, text in interpretations:
                    report_html += f"""
                    <div class="report-card" style="min-width: 250px; flex: 1;">
                        <span class="line-title">{name} Line</span><br>
                        <span class="interpretation">{text}</span>
                    </div>
                    """
                report_html += '</div>'
                report_placeholder.markdown(report_html, unsafe_allow_html=True)
            else:
                report_placeholder.info("Looking for palm lines...")

            if not run_stream: break
            time.sleep(0.01) # Small delay to prevent CPU choking

        cap.release()
    else:
        st.warning("Camera is inactive. Toggle 'Start Live Detection' to begin.")

else:
    # Upload Mode
    uploaded_file = st.file_uploader("Upload a photo of your palm", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        # Process and show
        img_cv2 = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        results = model.predict(source=img_cv2, conf=confidence_threshold, task='segment', save=False, verbose=False)
        result = results[0]
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB), use_column_width=True)
        
        with col2:
            st.subheader("Analysis Report")
            if result.masks is not None:
                masks = result.masks.xy
                class_ids = result.boxes.cls.cpu().numpy()
                for i, cid in enumerate(class_ids):
                    line_name = class_map.get(int(cid), "Unknown")
                    interp = analyze_line(masks[i], line_name)
                    st.markdown(f"""
                    <div class="report-card">
                        <span class="line-title">{line_name} Line</span><br>
                        <span class="interpretation">{interp}</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("No lines detected in the uploaded image.")

st.markdown("---")
st.caption("AI Palm Reader Pro | Powered by Ultralytics")
