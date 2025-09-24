import torch
import cv2
import numpy as np
from flask import Flask, request, jsonify
import io
from ultralytics import YOLO

# ================= Load YOLO Models =================
yolo_custom = YOLO("best.pt")       # custom YOLOv11
# yolo_nano = YOLO("yolov8n.pt")      # YOLOv8n bawaan
yolo_nano = YOLO("yolo11n.pt")      

YOLO_CUSTOM_CLASSES = yolo_custom.names
YOLO_NANO_CLASSES = yolo_nano.names

THRESHOLD = 0.4  # threshold untuk bounding box

# ================= Mapping =================
def map_to_category(label_name):
    kendaraan = ["car", "bus", "truck", "motorcycle", "bicycle", "truk", "mobil", "sepeda", "motor"]
    hewan = ["cat", "dog", "kucing", "anjing"]
    orang = ["person", "orang"]
    halangan = ["cone", "barrier", "bollard", "kursi", "pohon", "tiang", "gerobak"]

    if label_name in kendaraan:
        return "kendaraan"
    elif label_name in hewan:
        return "hewan"
    elif label_name in orang:
        return "orang"
    elif label_name in halangan:
        return "halangan"
    else:
        return "lainnya"

# ================= YOLO Detection =================
def detect_yolo_models(img_array):
    """Deteksi menggunakan YOLOv11n + custom model"""
    all_detections = []

    # YOLO custom
    results_custom = yolo_custom(img_array)[0]
    for box in results_custom.boxes:
        cls_id = int(box.cls.item())
        label_name = YOLO_CUSTOM_CLASSES[cls_id] if cls_id < len(YOLO_CUSTOM_CLASSES) else str(cls_id)
        all_detections.append({
            "model": "custom",
            "category": map_to_category(label_name),
            "label": label_name,
            "score": float(box.conf.item()),
            "xyxy": box.xyxy[0].cpu().numpy().tolist()  # ubah ke list biar bisa JSON
        })

    # YOLOv11n
    results_nano = yolo_nano(img_array)[0]
    for box in results_nano.boxes:
        cls_id = int(box.cls.item())
        label_name = YOLO_NANO_CLASSES[cls_id] if cls_id < len(YOLO_NANO_CLASSES) else str(cls_id)
        all_detections.append({
            "model": "yolov11n",
            "category": map_to_category(label_name),
            "label": label_name,
            "score": float(box.conf.item()),
            "xyxy": box.xyxy[0].cpu().numpy().tolist()  # ubah ke list
        })

    return all_detections

# ================= Filter Deteksi =================
def filter_detections(detections, threshold=THRESHOLD):
    """Hanya ambil deteksi dengan confidence >= threshold"""
    return [det for det in detections if det["score"] >= threshold]

# ================= Gambar Bounding Box =================
def draw_bboxes(img_array, detections, threshold=THRESHOLD):
    img_copy = img_array.copy()
    for det in detections:
        if det['score'] < threshold:
            continue
        x1, y1, x2, y2 = det["xyxy"]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        label = f"{det['label']} {det['score']:.2f}"
        color = (0,255,0) if det["model"]=="custom" else (255,0,0)
        cv2.rectangle(img_copy, (x1,y1), (x2,y2), color, 2)
        cv2.putText(img_copy, label, (x1,y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return img_copy

# ================= Flask App =================
app = Flask(__name__)

@app.route("/", methods=["POST"])
def upload_file():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    # Baca gambar
    in_memory_file = io.BytesIO()
    file.save(in_memory_file)
    data = np.frombuffer(in_memory_file.getvalue(), dtype=np.uint8)
    image_array = cv2.imdecode(data, cv2.IMREAD_COLOR)

    # Jalankan YOLO custom + YOLOv8n
    detections_yolo = detect_yolo_models(image_array)

    # Terapkan threshold
    detections_filtered = filter_detections(detections_yolo, THRESHOLD)

    # Hitung kategori
    category_counts = {}
    for det in detections_filtered:
        cat = det["category"]
        category_counts[cat] = category_counts.get(cat, 0) + 1

    # Buat summary
    summary_parts = [f"{count} {cat}" for cat, count in category_counts.items()]
    if not summary_parts:
        summary_text = "Terdeteksi ada objek asing di depan."
    elif len(summary_parts) == 1:
        summary_text = f"Terdeteksi {summary_parts[0]}"
    elif len(summary_parts) == 2:
        summary_text = f"Terdeteksi {summary_parts[0]} dan {summary_parts[1]}"
    else:
        summary_text = "Terdeteksi " + ", ".join(summary_parts[:-1]) + f", dan {summary_parts[-1]}"

    return jsonify({
        "summary": summary_text,
        "category_counts": category_counts,
        "detections": detections_filtered
    })

if __name__ == "__main__":
    app.run(debug=True)
