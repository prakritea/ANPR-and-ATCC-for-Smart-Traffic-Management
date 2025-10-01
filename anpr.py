# """
# lp_video_ocr.py
# YOLO (Ultralytics) + OCR pipeline for VIDEO input with simple IoU tracking + voting.
# Usage:
#   python lp_video_ocr.py --source path/to/video.mp4
#   python lp_video_ocr.py --source 0            # webcam
#   python lp_video_ocr.py --source path --weights best.pt --out out.mp4 --csv detections.csv
# """

# import cv2
# import numpy as np
# import easyocr
# import pytesseract
# import csv
# import time
# import argparse
# from ultralytics import YOLO
# from collections import deque, Counter
# import os

# # ---------------- config ----------------
# YOLO_WEIGHTS = "yolov8n.pt"  # Default weights (instead of best.pt)
# CONF_THRESH = 0.25
# IOU_THRESH = 0.3  # tracker matching IoU threshold
# OCR_LANGS = ['en']  # EasyOCR languages
# USE_EASYOCR = True
# OCR_EVERY_N_FRAMES = 3  # run OCR on each tracked object every N frames (helps speed)
# MIN_OCR_VOTES = 3  # minimum samples to produce a stable label
# MAX_MISSING_FRAMES = 12  # remove track if missing for this many frames
# OUTPUT_FPS = 20  # if saving video, output fps
# # ----------------------------------------

# # ---------------- utils ------------------
# def iou(boxA, boxB):
#     # boxes as (x1,y1,x2,y2)
#     xA = max(boxA[0], boxB[0])
#     yA = max(boxA[1], boxB[1])
#     xB = min(boxA[2], boxB[2])
#     yB = min(boxA[3], boxB[3])
#     interW = max(0, xB - xA)
#     interH = max(0, yB - yA)
#     interArea = interW * interH
#     if interArea == 0:
#         return 0.0
#     boxAArea = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
#     boxBArea = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
#     return interArea / float(boxAArea + boxBArea - interArea)

# def preprocess_for_ocr(plate_img):
#     # plate_img: BGR
#     gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
#     # resize to reasonable size
#     h,w = gray.shape
#     target_w = 400
#     if w < target_w:
#         scale = target_w / float(w)
#         gray = cv2.resize(gray, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_LINEAR)
#     gray = cv2.bilateralFilter(gray, 9, 75, 75)
#     th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                cv2.THRESH_BINARY, 11, 2)
#     return th

# def ocr_easyocr(reader, img):
#     # easyocr expects RGB or grayscale; return concatenated result
#     if len(img.shape) == 2:
#         img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
#     else:
#         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     try:
#         res = reader.readtext(img_rgb, detail=0, paragraph=False)
#         text = " ".join(res).strip()
#     except Exception as e:
#         text = ""
#     return text

# def ocr_tesseract(img):
#     if len(img.shape) == 3:
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     config = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
#     try:
#         t = pytesseract.image_to_string(img, config=config)
#         return t.strip()
#     except Exception:
#         return ""

# # Simple tracker using IoU matching and ID assignment
# class Track:
#     def __init__(self, tid, bbox, frame_idx):
#         self.id = tid
#         self.bbox = bbox  # current bbox (x1,y1,x2,y2)
#         self.last_seen = frame_idx
#         self.missing = 0
#         self.ocr_samples = []       # list of OCR strings collected
#         self.ocr_times = []         # corresponding frame indices / timestamps
#         self.confirmed_text = None  # final voted text when enough votes
#         self.age = 1

#     def update(self, bbox, frame_idx):
#         self.bbox = bbox
#         self.last_seen = frame_idx
#         self.missing = 0
#         self.age += 1

#     def mark_missing(self):
#         self.missing += 1

#     def add_ocr(self, text, frame_idx):
#         if text is None:
#             return
#         t = text.strip()
#         if t == "":
#             return
#         self.ocr_samples.append(t)
#         self.ocr_times.append(frame_idx)
#         # update confirmed_text if enough votes
#         if len(self.ocr_samples) >= MIN_OCR_VOTES:
#             c = Counter(self.ocr_samples)
#             most_common, cnt = c.most_common(1)[0]
#             # require majority
#             if cnt >= max(2, int(0.6 * len(self.ocr_samples))):
#                 self.confirmed_text = most_common

# # ---------------- main pipeline ----------------
# def process_video(source, weights, out_path=None, csv_path=None, use_easyocr=True):
#     # init models
#     model = YOLO(weights)
#     reader = None
#     if use_easyocr:
#         print("Initializing EasyOCR reader (this may take a few seconds)...")
#         reader = easyocr.Reader(OCR_LANGS, gpu=False)

#     cap = cv2.VideoCapture(source if str(source) != "0" else 0)
#     if not cap.isOpened():
#         raise RuntimeError(f"Cannot open source {source}")

#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     input_fps = cap.get(cv2.CAP_PROP_FPS) or OUTPUT_FPS
#     print(f"Input size: {width}x{height} FPS={input_fps:.2f}")

#     writer = None
#     if out_path:
#         fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#         writer = cv2.VideoWriter(out_path, fourcc, OUTPUT_FPS, (width, height))

#     csv_file = None
#     csv_writer = None
#     if csv_path:
#         csv_file = open(csv_path, "w", newline="", encoding="utf-8")
#         csv_writer = csv.writer(csv_file)
#         csv_writer.writerow(["timestamp_sec", "frame_idx", "track_id", "confimed_text", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"])

#     tracks = {}
#     next_track_id = 0
#     frame_idx = 0
#     t_start = time.time()

#     try:
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             frame_idx += 1
#             # Run YOLO detection on the frame
#             # Model.predict supports NumPy frames directly
#             results = model.predict(source=frame, conf=CONF_THRESH, iou=0.45, imgsz=1280, max_det=20, verbose=False)
#             if len(results) == 0:
#                 dets = []
#             else:
#                 r = results[0]
#                 dets = []
#                 if hasattr(r, "boxes") and r.boxes is not None:
#                     for b in r.boxes:
#                         xyxy = b.xyxy[0].cpu().numpy()
#                         x1,y1,x2,y2 = map(int, xyxy)
#                         conf = float(b.conf[0].cpu().numpy()) if hasattr(b, "conf") else None
#                         dets.append(((x1,y1,x2,y2), conf))

#             # Match detections to existing tracks by IoU
#             assigned_tracks = set()
#             assigned_dets = set()
#             # compute IoU matrix
#             if len(tracks) > 0 and len(dets) > 0:
#                 track_items = list(tracks.items())  # (tid, Track)
#                 iou_mat = np.zeros((len(track_items), len(dets)), dtype=float)
#                 for i, (tid, tr) in enumerate(track_items):
#                     for j, (bbox, conf) in enumerate(dets):
#                         iou_mat[i,j] = iou(tr.bbox, bbox)
#                 # greedy match highest IoU pairs
#                 while True:
#                     idx = np.unravel_index(np.argmax(iou_mat), iou_mat.shape)
#                     val = iou_mat[idx]
#                     if val <= IOU_THRESH:
#                         break
#                     ti, dj = idx
#                     tid = track_items[ti][0]
#                     if tid in assigned_tracks or dj in assigned_dets:
#                         iou_mat[ti, dj] = -1
#                         continue
#                     # assign
#                     tracks[tid].update(dets[dj][0], frame_idx)
#                     assigned_tracks.add(tid)
#                     assigned_dets.add(dj)
#                     iou_mat[ti, :] = -1
#                     iou_mat[:, dj] = -1

#             # create new tracks for unassigned detections
#             for j, (bbox, conf) in enumerate(dets):
#                 if j in assigned_dets:
#                     continue
#                 tid = next_track_id
#                 next_track_id += 1
#                 tracks[tid] = Track(tid, bbox, frame_idx)

#             # mark missing tracks and remove old ones
#             to_delete = []
#             for tid, tr in list(tracks.items()):
#                 if tr.last_seen != frame_idx:
#                     tr.mark_missing()
#                 if tr.missing > MAX_MISSING_FRAMES:
#                     # final write of confirmed_text if exists
#                     if csv_writer and tr.confirmed_text:
#                         t_sec = (time.time() - t_start)
#                         x1,y1,x2,y2 = tr.bbox
#                         csv_writer.writerow([f"{t_sec:.2f}", frame_idx, tid, tr.confirmed_text, x1,y1,x2,y2])
#                     to_delete.append(tid)
#             for tid in to_delete:
#                 del tracks[tid]

#             # OCR for tracks every N frames (or if track just created)
#             for tid, tr in tracks.items():
#                 run_ocr = False
#                 if tr.age <= 2:             # newly created
#                     run_ocr = True
#                 elif frame_idx % OCR_EVERY_N_FRAMES == 0:
#                     run_ocr = True
#                 if run_ocr:
#                     x1,y1,x2,y2 = tr.bbox
#                     # safety crop bounds
#                     x1c = max(0, x1); y1c = max(0, y1)
#                     x2c = min(frame.shape[1]-1, x2); y2c = min(frame.shape[0]-1, y2)
#                     if x2c - x1c <= 5 or y2c - y1c <= 5:
#                         continue
#                     crop = frame[y1c:y2c, x1c:x2c].copy()
#                     pre = preprocess_for_ocr(crop)
#                     if use_easyocr and reader is not None:
#                         text = ocr_easyocr(reader, pre)
#                     else:
#                         text = ocr_tesseract(pre)
#                     tr.add_ocr(text, frame_idx)

#             # Draw annotations
#             out = frame.copy()
#             for tid, tr in tracks.items():
#                 x1,y1,x2,y2 = tr.bbox
#                 color = (0,200,0) if tr.confirmed_text else (0,140,255)
#                 cv2.rectangle(out, (x1,y1), (x2,y2), color, 2)
#                 label = f"ID:{tid}"
#                 if tr.confirmed_text:
#                     label += f" {tr.confirmed_text}"
#                 elif len(tr.ocr_samples) > 0:
#                     # show most recent ocr sample
#                     label += f" {tr.ocr_samples[-1]}"
#                 cv2.putText(out, label, (x1, max(15, y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

#             # optionally write CSV rows for newly confirmed texts
#             if csv_writer:
#                 for tid, tr in tracks.items():
#                     # when a track just got a confirmed_text this frame, log it
#                     if tr.confirmed_text and (len(tr.ocr_times) > 0 and tr.ocr_times[-1] == frame_idx):
#                         t_sec = (time.time() - t_start)
#                         x1,y1,x2,y2 = tr.bbox
#                         csv_writer.writerow([f"{t_sec:.2f}", frame_idx, tid, tr.confirmed_text, x1,y1,x2,y2])

#             # display and write output
#             cv2.imshow("LP Video OCR", out)
#             if writer is not None:
#                 writer.write(out)

#             # quit key
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

#     finally:
#         cap.release()
#         if writer:
#             writer.release()
#         if csv_file:
#             csv_file.close()
#         cv2.destroyAllWindows()
#         print("Finished processing.")

# # --------------- CLI -----------------
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--source", "-s", required=True, help="Path to video file or '0' for webcam")
#     parser.add_argument("--weights", "-w", default=YOLO_WEIGHTS, help="YOLO weights path")
#     parser.add_argument("--out", "-o", default=None, help="Path to write annotated output video (mp4)")
#     parser.add_argument("--csv", default=None, help="Path to write detections CSV")
#     parser.add_argument("--no-easyocr", dest="easyocr", action="store_false", help="Use Tesseract instead of EasyOCR")
#     args = parser.parse_args()

#     src = args.source
#     if src == "0":
#         src = 0
#     process_video(src, args.weights, out_path=args.out, csv_path=args.csv, use_easyocr=args.easyocr)



# #to run
# #python -u "c:\Users\Prakriti\Desktop\New folder\anpr.py" --source "sample_video1.mp4" --weights "yolov8n.pt"









# import cv2
# import numpy as np
# import easyocr
# import pytesseract
# import csv
# from ultralytics import YOLO
# from collections import Counter

# # ---------------- config ----------------
# YOLO_WEIGHTS = "yolov8n.pt"  # Default weights
# CONF_THRESH = 0.25
# OCR_LANGS = ['en']  # EasyOCR languages
# USE_EASYOCR = True
# MIN_OCR_VOTES = 3  # minimum samples to produce a stable label
# OUTPUT_FPS = 20  # For video processing, but not used here
# # ----------------------------------------

# # ---------------- utils ------------------
# def iou(boxA, boxB):
#     xA = max(boxA[0], boxB[0])
#     yA = max(boxA[1], boxB[1])
#     xB = min(boxA[2], boxB[2])
#     yB = min(boxA[3], boxB[3])
#     interW = max(0, xB - xA)
#     interH = max(0, yB - yA)
#     interArea = interW * interH
#     if interArea == 0:
#         return 0.0
#     boxAArea = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
#     boxBArea = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
#     return interArea / float(boxAArea + boxBArea - interArea)

# def preprocess_for_ocr(plate_img):
#     gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
#     h, w = gray.shape
#     target_w = 400
#     if w < target_w:
#         scale = target_w / float(w)
#         gray = cv2.resize(gray, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_LINEAR)
#     gray = cv2.bilateralFilter(gray, 9, 75, 75)
#     th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
#     return th

# def ocr_easyocr(reader, img):
#     if len(img.shape) == 2:
#         img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
#     else:
#         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     try:
#         res = reader.readtext(img_rgb, detail=0, paragraph=False)
#         text = " ".join(res).strip()
#     except Exception as e:
#         text = ""
#     return text

# def ocr_tesseract(img):
#     if len(img.shape) == 3:
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     config = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
#     try:
#         t = pytesseract.image_to_string(img, config=config)
#         return t.strip()
#     except Exception:
#         return ""

# # Simple tracker using IoU matching and ID assignment
# class Track:
#     def __init__(self, tid, bbox):
#         self.id = tid
#         self.bbox = bbox  # current bbox (x1, y1, x2, y2)
#         self.ocr_samples = []  # List of OCR strings collected
#         self.confirmed_text = None  # Final voted text
#         self.age = 1

#     def update(self, bbox):
#         self.bbox = bbox
#         self.age += 1

#     def add_ocr(self, text):
#         if text is None or text.strip() == "":
#             return
#         self.ocr_samples.append(text.strip())
#         if len(self.ocr_samples) >= MIN_OCR_VOTES:
#             c = Counter(self.ocr_samples)
#             most_common, cnt = c.most_common(1)[0]
#             if cnt >= max(2, int(0.6 * len(self.ocr_samples))):
#                 self.confirmed_text = most_common

# # ---------------- main pipeline ----------------
# def process_image(image_path, weights, use_easyocr=True, output_csv=None):
#     # Initialize models
#     model = YOLO(weights)
#     reader = None
#     if use_easyocr:
#         print("Initializing EasyOCR reader (this may take a few seconds)...")
#         reader = easyocr.Reader(OCR_LANGS, gpu=False)

#     # Read the image
#     img = cv2.imread(image_path)
#     if img is None:
#         raise RuntimeError(f"Cannot open image {image_path}")

#     height, width, _ = img.shape
#     print(f"Image size: {width}x{height}")

#     # Run YOLO detection on the image
#     results = model.predict(source=img, conf=CONF_THRESH, iou=0.45, imgsz=640, max_det=10, verbose=False)
#     if len(results) == 0:
#         dets = []
#     else:
#         r = results[0]
#         dets = []
#         if hasattr(r, "boxes") and r.boxes is not None:
#             for b in r.boxes:
#                 xyxy = b.xyxy[0].cpu().numpy()
#                 x1, y1, x2, y2 = map(int, xyxy)
#                 dets.append(((x1, y1, x2, y2)))

#     tracks = {}
#     next_track_id = 0

#     # Create new tracks for each detected object (license plate)
#     for bbox in dets:
#         tid = next_track_id
#         next_track_id += 1
#         tracks[tid] = Track(tid, bbox)

#     # OCR for tracks
#     ocr_results = []
#     for tid, tr in tracks.items():
#         x1, y1, x2, y2 = tr.bbox
#         crop = img[y1:y2, x1:x2].copy()
#         pre = preprocess_for_ocr(crop)
#         text = ocr_easyocr(reader, pre) if use_easyocr else ocr_tesseract(pre)
#         tr.add_ocr(text)
#         print(f"Detected text: {text}")  # Debugging print statement
#         if tr.confirmed_text:
#             print(f"Confirmed Text: {tr.confirmed_text}")  # Debugging print statement
#             ocr_results.append([tr.confirmed_text, x1, y1, x2, y2])

#     # Save OCR results to CSV if specified
#     if output_csv:
#         with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
#             writer = csv.writer(file)
#             writer.writerow(["Detected Text", "x1", "y1", "x2", "y2"])
#             for result in ocr_results:
#                 writer.writerow(result)

#     # Draw annotations
#     out = img.copy()  # Copy the original image
#     for tid, tr in tracks.items():
#         x1, y1, x2, y2 = tr.bbox
#         color = (0, 200, 0) if tr.confirmed_text else (0, 140, 255)
#         cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
#         label = f"ID:{tid}"
#         if tr.confirmed_text:
#             label += f" {tr.confirmed_text}"
#         elif len(tr.ocr_samples) > 0:
#             label += f" {tr.ocr_samples[-1]}"
#         cv2.putText(out, label, (x1, max(15, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

#     # Display the result in the same size as the input image
#     cv2.imshow("LP Image OCR", out)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# # --------------- CLI -----------------
# if __name__ == "__main__":
#     image_path = "sample_img3.jpg"  # Replace with your image path
#     output_csv = "output_detected_plates.csv"  # CSV to store results
#     process_image(image_path, YOLO_WEIGHTS, use_easyocr=True, output_csv=output_csv)




import cv2
import numpy as np
import easyocr
import pytesseract
import csv
from ultralytics import YOLO
from collections import Counter

# ---------------- config ----------------
YOLO_WEIGHTS = "yolov8n.pt"  # Default weights
CONF_THRESH = 0.25
OCR_LANGS = ['en']  # EasyOCR languages
USE_EASYOCR = True
MIN_OCR_VOTES = 3  # minimum samples to produce a stable label
OUTPUT_FPS = 20  # For video processing, but not used here
# ----------------------------------------

# ---------------- utils ------------------
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    if interArea == 0:
        return 0.0
    boxAArea = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    boxBArea = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)

def preprocess_for_ocr(plate_img):
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    target_w = 400
    if w < target_w:
        scale = target_w / float(w)
        gray = cv2.resize(gray, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_LINEAR)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return th

def ocr_easyocr(reader, img):
    if len(img.shape) == 2:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    try:
        res = reader.readtext(img_rgb, detail=0, paragraph=False)
        text = " ".join(res).strip()
    except Exception as e:
        text = ""
    return text

def ocr_tesseract(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    config = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    try:
        t = pytesseract.image_to_string(img, config=config)
        return t.strip()
    except Exception:
        return ""

# Simple tracker using IoU matching and ID assignment
class Track:
    def __init__(self, tid, bbox):
        self.id = tid
        self.bbox = bbox  # current bbox (x1, y1, x2, y2)
        self.ocr_samples = []  # List of OCR strings collected
        self.confirmed_text = None  # Final voted text
        self.age = 1

    def update(self, bbox):
        self.bbox = bbox
        self.age += 1

    def add_ocr(self, text):
        if text is None or text.strip() == "":
            return
        # Normalize text (remove spaces and format)
        text = text.replace(" ", "").upper()  # Optional: Remove spaces and convert to uppercase
        self.ocr_samples.append(text.strip())
        if len(self.ocr_samples) >= MIN_OCR_VOTES:
            c = Counter(self.ocr_samples)
            most_common, cnt = c.most_common(1)[0]
            if cnt >= max(2, int(0.6 * len(self.ocr_samples))):
                self.confirmed_text = most_common

# ---------------- main pipeline ----------------
def process_image(image_path, weights, use_easyocr=True, output_csv=None):
    # Initialize models
    model = YOLO(weights)
    reader = None
    if use_easyocr:
        print("Initializing EasyOCR reader (this may take a few seconds)...")
        reader = easyocr.Reader(OCR_LANGS, gpu=False)

    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"Cannot open image {image_path}")

    height, width, _ = img.shape
    print(f"Image size: {width}x{height}")

    # Run YOLO detection on the image
    results = model.predict(source=img, conf=CONF_THRESH, iou=0.45, imgsz=640, max_det=10, verbose=False)
    if len(results) == 0:
        dets = []
    else:
        r = results[0]
        dets = []
        if hasattr(r, "boxes") and r.boxes is not None:
            for b in r.boxes:
                xyxy = b.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, xyxy)
                dets.append(((x1, y1, x2, y2)))

    tracks = {}
    next_track_id = 0

    # Create new tracks for each detected object (license plate)
    for bbox in dets:
        tid = next_track_id
        next_track_id += 1
        tracks[tid] = Track(tid, bbox)

    # OCR for tracks
    ocr_results = []
    for tid, tr in tracks.items():
        x1, y1, x2, y2 = tr.bbox
        crop = img[y1:y2, x1:x2].copy()
        pre = preprocess_for_ocr(crop)
        text = ocr_easyocr(reader, pre) if use_easyocr else ocr_tesseract(pre)
        tr.add_ocr(text)
        print(f"Detected text: {text}")  # Debugging print statement
        if tr.confirmed_text:
            print(f"Confirmed Text: {tr.confirmed_text}")  # Debugging print statement
            ocr_results.append([tr.confirmed_text, x1, y1, x2, y2])

    # Save OCR results to CSV if specified
    if output_csv:
        with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(["Detected Text", "x1", "y1", "x2", "y2"])
            for result in ocr_results:
                writer.writerow(result)

    # Draw annotations
    out = img.copy()  # Copy the original image
    for tid, tr in tracks.items():
        x1, y1, x2, y2 = tr.bbox
        color = (0, 200, 0) if tr.confirmed_text else (0, 140, 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"ID:{tid}"
        if tr.confirmed_text:
            label += f" {tr.confirmed_text}"
        elif len(tr.ocr_samples) > 0:
            label += f" {tr.ocr_samples[-1]}"
        cv2.putText(out, label, (x1, max(15, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Optional: Remove this part if you don't want to display the image
    # Display the result in the same size as the input image
    cv2.imshow("LP Image OCR", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# --------------- CLI -----------------
if __name__ == "__main__":
    image_path = "sample_img3.jpg"  # Replace with your image path
    output_csv = "output_detected_plates.csv"  # CSV to store results
    process_image(image_path, YOLO_WEIGHTS, use_easyocr=True, output_csv=output_csv)
