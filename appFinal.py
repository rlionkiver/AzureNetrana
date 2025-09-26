import torch
import cv2
import numpy as np
from flask import Flask, request, jsonify
import io
from ultralytics import YOLO
import paho.mqtt.client as mqtt
import json
import time
import logging
import random
import atexit

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ================= MQTT Configuration =================
MQTT_BROKER = "broker.emqx.io"
MQTT_PORT = 1883
MQTT_TOPIC_SOUND = "/netrana/sound"
MQTT_CLIENT_ID = f"flask_yolo_{int(time.time())}_{random.randint(1000, 9999)}"

# Global MQTT client
mqtt_client = None
mqtt_connected = False
mqtt_initialized = False

# =================  MQTT Setup =================
def setup_mqtt():
    global mqtt_client, mqtt_connected, mqtt_initialized
    
    if mqtt_initialized and mqtt_client is not None:
        logger.info("✅ MQTT already initialized, reusing existing connection")
        return True
        
    def on_connect(client, userdata, flags, rc):
        global mqtt_connected
        if rc == 0:
            logger.info("✅ MQTT Connected successfully to %s", MQTT_BROKER)
            mqtt_connected = True
        else:
            mqtt_connected = False
            error_codes = {
                1: "incorrect protocol version",
                2: "invalid client identifier",
                3: "server unavailable", 
                4: "bad username or password",
                5: "not authorised"
            }
            logger.error("❌ MQTT Connection failed: %s", error_codes.get(rc, f"code {rc}"))

    def on_disconnect(client, userdata, rc):
        global mqtt_connected
        mqtt_connected = False
        if rc != 0:
            logger.warning("⚠️ MQTT Unexpected disconnection. Reason: %s", rc)
            # Auto-reconnect will be handled by loop_start

    try:
        # Create client dengan configuration yang lebih robust
        mqtt_client = mqtt.Client(
            client_id=MQTT_CLIENT_ID,
            clean_session=True,
            protocol=mqtt.MQTTv311
        )
        
        # Set callback functions
        mqtt_client.on_connect = on_connect
        mqtt_client.on_disconnect = on_disconnect
        
        # Set last will testament (optional)
        mqtt_client.will_set(MQTT_TOPIC_SOUND, json.dumps({"status": "offline"}), qos=1)
        
        logger.info("🔗 Connecting to MQTT broker: %s:%d", MQTT_BROKER, MQTT_PORT)
        mqtt_client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
        
        # Start the loop
        mqtt_client.loop_start()
        mqtt_initialized = True
        logger.info("🚀 MQTT background thread started")
        
        # Wait for connection
        wait_time = 0
        while not mqtt_connected and wait_time < 10:
            time.sleep(0.5)
            wait_time += 0.5
            
        if mqtt_connected:
            logger.info("✅ MQTT setup completed successfully")
        else:
            logger.warning("⚠️ MQTT setup completed but not connected")
            
        return mqtt_connected
        
    except Exception as e:
        logger.error("❌ MQTT Setup failed: %s", e)
        mqtt_client = None
        return False

def cleanup_mqtt():
    """Cleanup MQTT connection on exit"""
    global mqtt_client, mqtt_connected
    if mqtt_client:
        logger.info("🛑 Cleaning up MQTT connection")
        mqtt_client.loop_stop()
        mqtt_client.disconnect()
        mqtt_connected = False

# Register cleanup function
atexit.register(cleanup_mqtt)

# ================= Load YOLO Models =================
yolo_custom = YOLO("best.pt")
yolo_nano = YOLO("yolo11n.pt")

YOLO_CUSTOM_CLASSES = yolo_custom.names
YOLO_NANO_CLASSES = yolo_nano.names

THRESHOLD = 0.4

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
            "xyxy": box.xyxy[0].cpu().numpy().tolist()
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
            "xyxy": box.xyxy[0].cpu().numpy().tolist()
        })

    return all_detections

def filter_detections(detections, threshold=THRESHOLD):
    return [det for det in detections if det["score"] >= threshold]

# ================= MQTT Message Sending =================
def send_mqtt_message(summary_text):
    global mqtt_client, mqtt_connected
    
    if not mqtt_connected:
        logger.warning("⚠️ MQTT not connected, attempting to send without reconnect")
        return False
    
    try:
        message_payload = {
            "summary": summary_text,
            "timestamp": time.time(),
            "type": "object_detection",
            "client_id": MQTT_CLIENT_ID
        }
        
        payload_json = json.dumps(message_payload)
        
        # Publish dengan QoS 0 untuk lebih cepat (tidak perlu acknowledgment)
        result = mqtt_client.publish(MQTT_TOPIC_SOUND, payload_json, qos=0)
        
        if result.rc == mqtt.MQTT_ERR_SUCCESS:
            logger.info("✅ MQTT Message sent to %s: %s", MQTT_TOPIC_SOUND, summary_text)
            return True
        else:
            logger.error("❌ MQTT Publish failed with code: %d", result.rc)
            return False
            
    except Exception as e:
        logger.error("❌ Error sending MQTT message: %s", e)
        return False

# ================= Flask App =================
app = Flask(__name__)

# Initialize MQTT once when module loads
logger.info("🚀 Initializing MQTT connection...")
mqtt_success = setup_mqtt()

@app.route("/", methods=["POST"])
def upload_file():
    global mqtt_connected
    
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    # Baca gambar
    in_memory_file = io.BytesIO()
    file.save(in_memory_file)
    data = np.frombuffer(in_memory_file.getvalue(), dtype=np.uint8)
    image_array = cv2.imdecode(data, cv2.IMREAD_COLOR)

    # Jalankan deteksi
    detections_yolo = detect_yolo_models(image_array)
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

    # Kirim ke MQTT hanya jika terhubung
    mqtt_success = False
    if mqtt_connected:
        mqtt_success = send_mqtt_message(summary_text)
    else:
        logger.warning("⚠️ Skipping MQTT send - not connected")
    
    return jsonify({
        "summary": summary_text,
        "mqtt_sent": mqtt_success,
        "mqtt_connected": mqtt_connected,
        "mqtt_topic": MQTT_TOPIC_SOUND,
        "objects_detected": category_counts
    })

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "mqtt_connected": mqtt_connected,
        "mqtt_broker": MQTT_BROKER,
        "mqtt_port": MQTT_PORT,
        "mqtt_client_id": MQTT_CLIENT_ID
    })

if __name__ == "__main__":
    app.run(debug=True) 