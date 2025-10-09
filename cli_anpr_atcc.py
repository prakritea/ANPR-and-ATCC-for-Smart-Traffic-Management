import cv2
import csv
from datetime import datetime
from collections import defaultdict
from ultralytics import YOLO
import easyocr
import re


def preprocess_for_ocr(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return th


def ocr_easyocr(reader, img):
    pre = preprocess_for_ocr(img)
    img_rgb = cv2.cvtColor(pre, cv2.COLOR_GRAY2RGB)
    try:
        result = reader.readtext(img_rgb, detail=0, paragraph=False)
        return " ".join(result).strip()
    except Exception:
        return ""


def is_valid_plate(text: str):
    text = text.replace(" ", "").upper()
    return bool(re.match(r"^[A-Z0-9]{4,10}$", text))


def crop_plate_region(vehicle_img):
    h = vehicle_img.shape[0]
    if h < 10:
        return None
    return vehicle_img[int(h * 0.75) :, :]


def run(video_path: str, weights: str = "yolov8n.pt", out_csv: str = "vehicle_log.csv"):
    model = YOLO(weights)
    reader = easyocr.Reader(["en"], gpu=False)
    class_list = model.names

    cap = cv2.VideoCapture(video_path)
    crossed_ids = set()
    plate_numbers = {}
    vehicle_counts = defaultdict(int)
    ocr_attempts = defaultdict(int)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Vehicle Type", "Plate", "Timestamp"]) 

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model.track(frame, persist=True, classes=[2, 3, 5, 7])
            if not results or not results[0].boxes:
                continue

            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy()

            for box, track_id, cls_id in zip(boxes, ids, class_ids):
                x1, y1, x2, y2 = map(int, box)
                class_name = class_list[int(cls_id)]

                if (x2 - x1) * (y2 - y1) < 1500:
                    continue

                # Count once per id when it crosses mid-height
                cy = (y1 + y2) // 2
                if cy > frame.shape[0] // 2 and track_id not in crossed_ids:
                    crossed_ids.add(track_id)
                    vehicle_counts[class_name] += 1

                # Try OCR a few times per id
                if track_id not in plate_numbers and ocr_attempts[track_id] < 3:
                    vehicle_crop = frame[y1:y2, x1:x2]
                    plate_crop = crop_plate_region(vehicle_crop)
                    if plate_crop is not None and plate_crop.size > 0:
                        text = ocr_easyocr(reader, plate_crop).upper().replace(" ", "")
                        if is_valid_plate(text):
                            plate_numbers[track_id] = text
                            writer.writerow([int(track_id), class_name, text, datetime.now().isoformat(timespec="seconds")])
                        ocr_attempts[track_id] += 1

    cap.release()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True, help="Path to input video")
    p.add_argument("--weights", default="yolov8n.pt")
    p.add_argument("--out", default="vehicle_log.csv")
    args = p.parse_args()
    run(args.video, weights=args.weights, out_csv=args.out)


