import streamlit as st
import tempfile
import time
import pandas as pd
from ultralytics import YOLO
from utils.analytics import traffic_density, TrafficHistory

# -------------------------------
# Page configuration
# -------------------------------
st.set_page_config(
    page_title="Smart City Traffic Surveillance",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------
# Light UI styling
# -------------------------------
st.markdown("""
<style>
body { background-color: #f5f7fb; }
.block-container { padding-top: 1.2rem; padding-left: 2rem; padding-right: 2rem; }

.metric-card {
    background: #ffffff;
    border-radius: 12px;
    padding: 18px;
    text-align: center;
    box-shadow: 0 6px 16px rgba(0,0,0,0.08);
}

.alert-high   { background:#fee2e2; color:#991b1b; padding:10px; border-radius:8px; font-weight:600; }
.alert-medium { background:#fef3c7; color:#92400e; padding:10px; border-radius:8px; font-weight:600; }
.alert-low    { background:#dcfce7; color:#166534; padding:10px; border-radius:8px; font-weight:600; }

.video-box {
    border-radius: 14px;
    overflow: hidden;
    background: white;
    box-shadow: 0 8px 22px rgba(0,0,0,0.08);
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Sidebar controls
# -------------------------------
with st.sidebar:
    st.header("🎛️ Detection Controls")

    mode = st.selectbox("Detection Mode", ["Balanced", "High Accuracy", "High Performance"])
    if mode == "High Accuracy":
        confidence_default, iou_default, frame_skip_default = 0.4, 0.5, 1
    elif mode == "High Performance":
        confidence_default, iou_default, frame_skip_default = 0.25, 0.4, 3
    else:
        confidence_default, iou_default, frame_skip_default = 0.3, 0.45, 2

    model_choice = st.selectbox("YOLO Model", ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"], index=1)

    confidence = st.slider("Confidence Threshold", 0.1, 0.9, confidence_default, 0.05)
    iou_thres  = st.slider("IoU Threshold", 0.1, 0.9, iou_default, 0.05)
    frame_skip = st.slider("Frame Skip", 1, 6, frame_skip_default)

    resolution = st.selectbox("Resolution", ["640x360", "854x480", "1280x720"])
    width, height = map(int, resolution.split("x"))

    detect_vehicles = st.checkbox("Detect Vehicles", True)
    detect_persons  = st.checkbox("Detect Persons", True)

    st.markdown("### 🚨 Congestion Thresholds")
    high_thresh   = st.number_input("High Congestion (vehicles)", value=25, min_value=5)
    medium_thresh = st.number_input("Medium Congestion (vehicles)", value=10, min_value=1)

# -------------------------------
# Load YOLO model
# -------------------------------
@st.cache_resource
def load_model(name):
    return YOLO(f"models/{name}")

model = load_model(model_choice)

VEHICLES = ["car", "bus", "truck", "motorcycle"]
PERSON   = "person"

# -------------------------------
# Header
# -------------------------------
st.title("🚦 Smart City Traffic Monitoring System")
st.caption("AI-powered traffic monitoring with real-time congestion alerts using YOLOv8")

# -------------------------------
# Congestion alert logic
# -------------------------------
def congestion_alert(vehicle_count):
    if vehicle_count >= high_thresh:
        return "HIGH"
    elif vehicle_count >= medium_thresh:
        return "MEDIUM"
    return "LOW"

# -------------------------------
# Dashboard rendering
# -------------------------------
def render_dashboard(persons, vehicles, density, alert, history):
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("👤 Persons", persons)
    c2.metric("🚗 Vehicles", vehicles)
    c3.metric("📊 Density", density)
    c4.metric("🏆 Peak Vehicles", history.peak_vehicles)
    c5.metric("🚨 Alert", alert)

# -------------------------------
# Activity chart
# -------------------------------
def render_chart(history):
    data = history.as_chart_data()
    if len(data) > 1:
        df = pd.DataFrame(data).set_index("time")
        st.line_chart(df, use_container_width=True, height=150)

# -------------------------------
# Frame detection
# -------------------------------
def detect_frame(frame):
    frame_small = cv2.resize(frame, (width, height))
    results = model.predict(frame_small, conf=confidence, iou=iou_thres, verbose=False)
    annotated = results[0].plot()

    vehicles = 0
    persons = 0
    for box in results[0].boxes:
        label = model.names[int(box.cls[0])]
        if label in VEHICLES and detect_vehicles:
            vehicles += 1
        elif label == PERSON and detect_persons:
            persons += 1

    return annotated, vehicles, persons

# -------------------------------
# Stream processing (Video or Webcam)
# -------------------------------
def run_stream(cap, history_size):
    history = TrafficHistory(maxlen=history_size)
    left, right = st.columns([3, 2])
    frame_box = left.empty()
    dashboard = right.empty()
    chart_area = right.empty()
    frame_index = 0

    while cap.isOpened():
        if st.session_state.get("stop_cam", False):
            break
        ret, frame = cap.read()
        if not ret:
            break

        frame_index += 1
        if frame_index % frame_skip != 0:
            continue

        annotated, vehicles, persons = detect_frame(frame)
        history.record(vehicles, persons)
        density = traffic_density(vehicles)
        alert = congestion_alert(vehicles)

        frame_box.image(annotated, channels="BGR", use_container_width=True)
        with dashboard.container():
            render_dashboard(persons, vehicles, density, alert, history)
        with chart_area.container():
            render_chart(history)

        time.sleep(0.02)

    cap.release()

# -------------------------------
# Video analysis
# -------------------------------
st.subheader("📹 Video Traffic Analysis")
video_file = st.file_uploader("Upload traffic footage", type=["mp4","avi","mov"])
if video_file:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(video_file.read())
    tfile.flush()
    cap = cv2.VideoCapture(tfile.name)
    run_stream(cap, history_size=120)

# -------------------------------
# Live camera monitoring
# -------------------------------
st.subheader("📸 Live Camera Monitoring")
col1, col2 = st.columns(2)

with col1:
    if st.button("Start Camera", use_container_width=True):
        st.session_state["stop_cam"] = False
        cap = cv2.VideoCapture(0)
        run_stream(cap, history_size=60)

with col2:
    if st.button("Stop Camera", use_container_width=True):
        st.session_state["stop_cam"] = True
        st.info("Camera stopped.")
