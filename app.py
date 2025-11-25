import streamlit as st
import cv2
import time
import pandas as pd
import plotly.express as px
from ultralytics import YOLO
import tempfile
import os

# ================= CONFIGURATION =================
st.set_page_config(
    page_title="Autonomous Vehicle Monitor",
    page_icon="🚗",
    layout="wide"
)

# --- THE BAN LIST (Same as before) ---
CLASS_RULES = {
    1: 0.30, 2: 0.50, 3: 0.50, 11: 0.30, 12: 0.30, 13: 0.30, 
    14: 0.40, 15: 0.20, 16: 0.50, 17: 0.50,
    0: 2.0, 4: 2.0, 5: 2.0, 6: 2.0, 7: 2.0, 8: 2.0, 9: 2.0, 10: 2.0
}
DEFAULT_CONF = 0.50 

# ================= HELPER FUNCTIONS =================
@st.cache_resource
def load_model(model_path):
    """Load model once and cache it to save memory"""
    return YOLO(model_path, task='detect')

def process_video(video_path, model_path, conf_threshold, img_size):
    model = load_model(model_path)
    cap = cv2.VideoCapture(video_path)
    
    # Dashboard Layout Containers
    st.markdown("### 📡 Live Telemetry Feed")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    col1, col2 = st.columns([2, 1]) # Video gets 2/3 width
    with col1:
        image_spot = st.empty()
    with col2:
        chart_spot_lat = st.empty()
        chart_spot_obj = st.empty()

    data_buffer = []
    frame_id = 0
    
    while cap.isOpened():
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_id += 1
        
        # Optimization: Skip frames if video is high FPS to keep UI responsive
        if frame_id % 3 != 0: 
            continue

        # 1. Inference
        # Force CPU for Cloud Compatibility
        results = model(frame, imgsz=img_size, conf=0.1, device='cpu', verbose=False)
        
        # 2. Filter
        final_boxes = []
        if len(results) > 0:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                threshold = CLASS_RULES.get(cls_id, DEFAULT_CONF)
                if conf >= threshold:
                    final_boxes.append(box)
            results[0].boxes = final_boxes

        # 3. Stats
        end_time = time.time()
        inference_time = end_time - start_time
        fps = 1 / inference_time if inference_time > 0 else 0
        latency_ms = inference_time * 1000
        objects_detected = len(final_boxes)

        # 4. Update Data
        data_buffer.append({
            "Frame": frame_id,
            "FPS": round(fps, 1),
            "Latency_ms": round(latency_ms, 1),
            "Objects": objects_detected
        })
        
        # Keep buffer small for speed
        if len(data_buffer) > 50: 
            data_buffer.pop(0)
        df = pd.DataFrame(data_buffer)

        # 5. Update UI (Real-time)
        kpi1.metric("System Health", "ONLINE")
        kpi2.metric("Inference Speed", f"{fps:.1f} FPS")
        kpi3.metric("Latency", f"{latency_ms:.1f} ms")
        kpi4.metric("Objects Detected", f"{objects_detected}")

        # Draw Video
        annotated_frame = results[0].plot()
        # Convert BGR to RGB for Streamlit
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        image_spot.image(annotated_frame, caption=f"Real-Time Inference (Frame {frame_id})", use_container_width=True)

        # Draw Charts
        if not df.empty:
            fig_lat = px.line(df, x="Frame", y="Latency_ms", title="Latency Stability", height=250)
            fig_lat.update_layout(margin=dict(l=20, r=20, t=30, b=20))
            chart_spot_lat.plotly_chart(fig_lat, use_container_width=True, key=f"lat_{frame_id}")

            fig_obj = px.bar(df, x="Frame", y="Objects", title="Object Density", height=250)
            fig_obj.update_layout(margin=dict(l=20, r=20, t=30, b=20))
            chart_spot_obj.plotly_chart(fig_obj, use_container_width=True, key=f"obj_{frame_id}")

    cap.release()

# ================= MAIN APP LOGIC =================
st.title("🚗 Autonomous Vehicle MLOps Platform")
st.sidebar.header("Deployment Settings")

# 1. Model Selection
model_file = "best.onnx"
if not os.path.exists(model_file):
    st.error(f"Model file '{model_file}' not found! Please upload it to your GitHub repo.")
    st.stop()

# 2. Input Source
source_type = st.sidebar.radio("Select Input Source", ["Sample Video", "Upload Video"])

if source_type == "Sample Video":
    video_file = "Relaxing Night Drive in Tokyo _ 8K 60fps HDR _ Soft Lofi Beats - Abao Vision (1080p, h264).mp4" 
    if not os.path.exists(video_file):
        st.warning("Sample video not found in repository.")
        video_file = None
else:
    uploaded_file = st.sidebar.file_uploader("Upload a driving video", type=['mp4', 'mov', 'avi'])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_file = tfile.name
    else:
        video_file = None

# 3. Start Button
if st.sidebar.button("🚀 Start Deployment Test"):
    if video_file:
        process_video(video_file, model_file, 0.5, 320)
    else:
        st.error("Please select or upload a video first.")
else:
    st.info("👈 Select settings in the sidebar and click 'Start Deployment Test'")