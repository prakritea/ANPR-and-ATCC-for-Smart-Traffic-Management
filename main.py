import cv2
from ultralytics import YOLO
import easyocr
import numpy as np
from collections import defaultdict
import re
import csv
import streamlit as st
from datetime import datetime
import tempfile
import sys
import io

# ---------------- STREAMLIT CONFIG ----------------
st.set_page_config(page_title="🚦 Smart Traffic System (ANPR + ATCC)", layout="wide")

st.title("🚦 Smart Traffic System (ANPR + ATCC)")

with st.expander("ℹ️ About Project", expanded=True):
    st.write("""
    This project integrates **Automatic Vehicle Counting (ATCC)** and 
    **Automatic Number Plate Recognition (ANPR)** into a single system.  
    - Detects vehicles using **YOLOv8**  
    - Reads license plates using **EasyOCR**  
    - Counts each category of vehicles crossing a line  
    - Logs detected vehicles with timestamp into a **CSV file**  
    - Displays live results, logs, and speed metrics in real-time  
    """)

# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙️ Settings")

line_y = st.sidebar.slider("Counting Line Y", 100, 800, 430)
use_gpu = st.sidebar.checkbox("Use GPU for OCR", False)
min_box_area = st.sidebar.slider("Minimum Box Area", 500, 10000, 2000, step=100)

uploaded_video = st.sidebar.file_uploader("📤 Upload a video file", type=["mp4", "avi", "mov"])

# ---------------- PLACEHOLDERS ----------------
video_placeholder = st.empty()
caption_placeholder = st.empty()
stats_placeholder = st.empty()
log_placeholder = st.empty()
speed_placeholder = st.empty()
console_placeholder = st.container()

# ---------------- LOGGER ----------------
class StreamlitLogger(io.StringIO):
    def write(self, msg):
        if msg.strip():
            with console_placeholder:
                st.markdown(
                    f"<div style='background-color:#ffdddd;padding:5px;border-radius:5px;'>{msg}</div>",
                    unsafe_allow_html=True,
                )
        return super().write(msg)

sys.stderr = StreamlitLogger()
sys.stdout = StreamlitLogger()

# ---------------- HELPER FUNCTIONS ----------------
def preprocess_for_ocr(plate_img):
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    return thresh


def ocr_easyocr(reader, img):
    img_pre = preprocess_for_ocr(img)
    img_rgb = cv2.cvtColor(img_pre, cv2.COLOR_GRAY2RGB)
    try:
        result = reader.readtext(img_rgb, detail=0, paragraph=False)
        return " ".join(result).strip()
    except:
        return ""


def is_valid_plate(text):
    text = text.replace(" ", "")
    return bool(re.match(r"^[A-Z0-9]{4,10}$", text))


def crop_plate_region(vehicle_img):
    h = vehicle_img.shape[0]
    if h < 10:
        return None
    return vehicle_img[int(h * 0.75) :, :]


# ---------------- MAIN PROCESS ----------------
if uploaded_video is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())

    st.sidebar.success("✅ Video uploaded. Click 'Start Processing' below.")
    run_button = st.sidebar.button("▶️ Start Processing")

    if run_button:
        st.sidebar.info("🚀 Processing started...")

        model = YOLO("yolo11l.pt")
        reader = easyocr.Reader(["en"], gpu=use_gpu)
        class_list = model.names

        cap = cv2.VideoCapture(tfile.name)

        crossed_ids = set()
        plate_numbers = {}
        vehicle_counts = defaultdict(int)
        ocr_attempts = defaultdict(int)
        logs = []

        output_csv = "vehicle_log.csv"
        csv_file = open(output_csv, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["ID", "Vehicle Type", "Plate", "Timestamp"])

        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model.track(frame, persist=True, classes=[2, 3, 5, 7])

            if not results or not results[0].boxes:
                video_placeholder.image(frame, channels="BGR")
                continue

            # Extract detection boxes and IDs
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy()

            # Draw counting line
            cv2.line(frame, (100, line_y), (frame.shape[1] - 100, line_y), (0, 0, 255), 2)

            for box, track_id, cls_id in zip(boxes, ids, class_ids):
                x1, y1, x2, y2 = map(int, box)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                class_name = class_list[int(cls_id)]

                if (x2 - x1) * (y2 - y1) < min_box_area:
                    continue

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"ID:{int(track_id)} {class_name}"
                if track_id in plate_numbers:
                    label += f" | Plate: {plate_numbers[track_id]}"
                cv2.putText(
                    frame,
                    label,
                    (x1, max(30, y1 - 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2,
                )

                # Counting
                if cy > line_y and track_id not in crossed_ids:
                    crossed_ids.add(track_id)
                    vehicle_counts[class_name] += 1

                # OCR
                if track_id not in plate_numbers and ocr_attempts[track_id] < 3:
                    vehicle_crop = frame[y1:y2, x1:x2]
                    plate_crop = crop_plate_region(vehicle_crop)
                    if plate_crop is not None and plate_crop.size > 0:
                        text = ocr_easyocr(reader, plate_crop).upper().replace(" ", "")
                        if is_valid_plate(text):
                            plate_numbers[track_id] = text
                            timestamp = datetime.now().strftime("%H:%M:%S")
                            logs.append([track_id, class_name, text, timestamp])
                            csv_writer.writerow([track_id, class_name, text, timestamp])
                            caption_placeholder.info(f"📌 Detected {class_name} | Plate: {text}")
                        ocr_attempts[track_id] += 1

            # Show video frame
            video_placeholder.image(frame, channels="BGR")

            # Retrieve YOLO speed stats
            speed_stats = results[0].speed  # dict with preprocess, inference, postprocess

            # Update dashboard every 20 frames
            if frame_count % 20 == 0:
                with stats_placeholder:
                    st.subheader("📊 Vehicle Counts")
                    cols = st.columns(4)
                    for i, (name, count) in enumerate(vehicle_counts.items()):
                        cols[i % 4].metric(label=name.capitalize(), value=count)

                with speed_placeholder:
                    st.subheader("⚡ YOLO Processing Speed (ms)")
                    s_cols = st.columns(3)
                    s_cols[0].metric("Preprocess", f"{speed_stats['preprocess']:.2f} ms")
                    s_cols[1].metric("Inference", f"{speed_stats['inference']:.2f} ms")
                    s_cols[2].metric("Postprocess", f"{speed_stats['postprocess']:.2f} ms")

                with log_placeholder:
                    st.subheader("📝 Vehicle Log")
                    if logs:
                        st.dataframe(logs, width="stretch")

            frame_count += 1

        cap.release()
        csv_file.close()
        st.sidebar.success("✅ Processing completed!")

        with open(output_csv, "rb") as f:
            st.download_button(
                "⬇️ Download CSV Log", f, file_name=output_csv, mime="text/csv"
            )
