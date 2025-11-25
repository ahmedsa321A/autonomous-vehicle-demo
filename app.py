import streamlit as st
import cv2
import time
import pandas as pd
import plotly.express as px
from ultralytics import YOLO
import tempfile
import os
import numpy as np
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# ================= CONFIGURATION =================
st.set_page_config(
    page_title="Autonomous Vehicle Monitor",
    page_icon="🚗",
    layout="wide"
)

# --- THE BAN LIST ---
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

def process_frame(frame, model, conf_threshold, img_size):
    """Run inference and return results + object counts"""
    # 1. Inference
    results = model(frame,  conf=conf_threshold, device='cpu', verbose=False)
    
    # 2. Filter & Count
    final_boxes = []
    counts = {}
    
    if len(results) > 0:
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            threshold = CLASS_RULES.get(cls_id, DEFAULT_CONF)
            
            if conf >= threshold:
                final_boxes.append(box)
                # Add to counts
                class_name = model.names[cls_id]
                counts[class_name] = counts.get(class_name, 0) + 1
                
        results[0].boxes = final_boxes
    
    return results, counts

def draw_hud(frame, fps, latency_ms, counts):
    """Draw stats and class breakdown on the frame"""
    total_obj = sum(counts.values())
    
    # Line 1: Technical Stats
    stats_text = f"FPS: {fps:.1f} | Latency: {latency_ms:.1f}ms | Total: {total_obj}"
    
    # Line 2: Object Breakdown (e.g., "car: 3 | person: 1")
    breakdown_text = " | ".join([f"{k}: {v}" for k, v in counts.items()])
    
    # Draw Black Background
    # Height adjusts if we have a second line of text
    h_bg = 70 if breakdown_text else 40
    cv2.rectangle(frame, (0, 0), (650, h_bg), (0, 0, 0), -1)
    
    # Draw Text (Green for stats, Yellow for objects)
    cv2.putText(frame, stats_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    if breakdown_text:
        cv2.putText(frame, breakdown_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
    return frame

# --- VIDEO PROCESSOR CLASS (Fixes Threading Issues) ---
class YOLOVideoProcessor:
    def __init__(self):
        # Initialize model and timers
        self.model = load_model("best.onnx")
        self.last_time = time.time()

    def recv(self, frame):
        try:
            start_time = time.time()
            img = frame.to_ndarray(format="bgr24")
            
            # Process
            results, counts = process_frame(img, self.model, 0.5, 320)
            
            # Draw Boxes
            annotated_frame = results[0].plot()
            
            # --- CALCULATE STATS ---
            end_time = time.time()
            inference_time = end_time - start_time
            total_time = end_time - self.last_time
            fps = 1 / total_time if total_time > 0 else 0
            latency_ms = inference_time * 1000
            self.last_time = end_time

            # --- DRAW HUD ---
            annotated_frame = draw_hud(annotated_frame, fps, latency_ms, counts)

            # Return back to browser
            return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")
        except Exception as e:
            print(f"Error processing frame: {e}")
            return frame

def process_video(video_path, model_path, conf_threshold, img_size):
    model = load_model(model_path)
    cap = cv2.VideoCapture(video_path)
    
    st.markdown("### 📡 Live Telemetry Feed")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    col1, col2 = st.columns([2, 1])
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
        if frame_id % 3 != 0: continue # Skip frames for speed

        # Run shared processing logic
        results, counts = process_frame(frame, model, conf_threshold, img_size)
        objects_detected = sum(counts.values())

        # Stats
        end_time = time.time()
        inference_time = end_time - start_time
        fps = 1 / inference_time if inference_time > 0 else 0
        latency_ms = inference_time * 1000

        # Update Data
        data_buffer.append({
            "Frame": frame_id,
            "FPS": round(fps, 1),
            "Latency_ms": round(latency_ms, 1),
            "Objects": objects_detected
        })
        
        if len(data_buffer) > 50: data_buffer.pop(0)
        df = pd.DataFrame(data_buffer)

        # UI Updates
        kpi1.metric("System Health", "ONLINE")
        kpi2.metric("Inference Speed", f"{fps:.1f} FPS")
        kpi3.metric("Latency", f"{latency_ms:.1f} ms")
        kpi4.metric("Objects Detected", f"{objects_detected}")

        # Draw HUD on frame
        annotated_frame = results[0].plot()
        annotated_frame = draw_hud(annotated_frame, fps, latency_ms, counts)
        
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        image_spot.image(annotated_frame, caption=f"Real-Time Inference (Frame {frame_id})", use_container_width=True)

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

model_file = "best.onnx"
if not os.path.exists(model_file):
    st.error(f"Model file '{model_file}' not found! Please upload it to your GitHub repo.")
    st.stop()

# --- CONFIDENCE SLIDER ---
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.50, 0.05)

# 1. Source Selection
source_type = st.sidebar.radio("Select Input Source", ["Sample Video", "Upload Video", "Live Stream (WebRTC)", "Snapshot (Legacy)"])

if source_type == "Live Stream (WebRTC)":
    st.markdown("### 📡 Real-Time Camera Stream")
    st.info("Click 'Start' and allow camera access. Works on Mobile & Desktop.")
    
    # WebRTC Configuration (Google STUN server ensures it works on mobile networks)
    rtc_configuration = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
    
    webrtc_streamer(
        key="yolo-stream",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_configuration,
        video_processor_factory=YOLOVideoProcessor,  # Using Class-based processor for stability
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

elif source_type == "Snapshot (Legacy)":
    st.markdown("### 📸 Phone Camera Snapshot")
    camera_image = st.camera_input("Tap to Capture")
    if camera_image:
        file_bytes = np.asarray(bytearray(camera_image.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)
        
        model = load_model(model_file)
        start_time = time.time()
        results, counts = process_frame(frame, model, conf_threshold, 320)
        end_time = time.time()
        
        latency_ms = (end_time - start_time) * 1000
        fps = 0 # Snapshot doesn't have meaningful FPS
        
        annotated_frame = results[0].plot()
        annotated_frame = draw_hud(annotated_frame, fps, latency_ms, counts)
        
        st.image(annotated_frame, caption="Processed Snapshot", channels="BGR", use_container_width=True)

elif source_type == "Sample Video":
    video_file = "Relaxing Night Drive in Tokyo _ 8K 60fps HDR _ Soft Lofi Beats - Abao Vision (1080p, h264).mp4"
    if st.sidebar.button("🚀 Start Sample"):
        if os.path.exists(video_file):
            process_video(video_file, model_file, conf_threshold, 320)
        else:
            st.warning("Sample video not found.")

else: # Upload Video
    uploaded_file = st.sidebar.file_uploader("Upload a driving video", type=['mp4', 'mov', 'avi'])
    if uploaded_file and st.sidebar.button("🚀 Start Processing"):
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        process_video(tfile.name, model_file, conf_threshold, 320)