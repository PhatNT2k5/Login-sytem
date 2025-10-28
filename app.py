import cv2
import numpy as np
from flask import Flask, render_template, Response
from PIL import Image

from face_detector import FaceDetectorYuNet, FaceDetectorHaar
from liveness_detector import LivenessFAS
from face_recognizer import FaceRecognizer

YUNET_MODEL = "model/face_detection_yunet_2023mar.onnx"
LIVENESS_MODEL = "model/liveness_fas_in224.onnx"
YOLO_CLS_MODEL = "model/yolo-classification.pt"
FACE_DB = "model/face_db.pkl"

LIVE_MIN_SCORE = 0.495  # ngưỡng trên prob_real

app = Flask(__name__)

# load models
try:
    try:
        face_detector = FaceDetectorYuNet(model_path=YUNET_MODEL)
    except Exception:
        face_detector = FaceDetectorHaar()

    # chọn chế độ preprocess tương thích training (đề xuất: "div255_only")
    liveness_model = LivenessFAS(model_path=LIVENESS_MODEL, preprocess_mode="div255_only")

    face_recognizer = FaceRecognizer(
        model_path=YOLO_CLS_MODEL,
        db_path=FACE_DB,
        device="cpu",
        threshold=0.7,
    )
except Exception as e:
    print(f"[ERROR] Không thể tải mô hình: {e}")
    raise

def _expand_box(x, y, w, h, W, H, margin_ratio=0.12):  # giảm margin mặc định
    dw = int(w * margin_ratio)
    dh = int(h * margin_ratio)
    x1 = max(0, x - dw); y1 = max(0, y - dh)
    x2 = min(W, x + w + dw); y2 = min(H, y + h + dh)
    return x1, y1, x2, y2

def generate_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Không mở được webcam.")
        return

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        H, W = frame.shape[:2]
        faces = face_detector.detect(frame) if isinstance(face_detector, FaceDetectorYuNet) \
            else face_detector.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

        for (x, y, w, h) in faces:
            x1, y1, x2, y2 = _expand_box(x, y, w, h, W, H, margin_ratio=0.12)
            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue

            try:
                # 1) Liveness
                live_label, prob_real = liveness_model.predict(face_crop)

                # CHỈ so trên prob_real (đã softmax/sigmoid), không AND với label string
                is_live = (prob_real >= LIVE_MIN_SCORE)

                if not is_live:
                    disp_name = "Unknown"
                    id_score = 0.0
                    info_label = f"SPOOF ({prob_real:.2f})"
                    box_color = (0, 0, 255)
                else:
                    pil_img = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
                    name, id_score = face_recognizer.recognize(pil_img)
                    disp_name = name if id_score >= face_recognizer.threshold else "Unknown"
                    info_label = f"{disp_name} ({id_score:.2f})"
                    box_color = (0, 255, 0)

                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                text = f"{info_label} - Liveness:{prob_real:.2f}"
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                ty1 = max(0, y1 - 10 - th)
                cv2.rectangle(frame, (x1, ty1), (x1 + tw + 6, ty1 + th + 6), box_color, -1)
                cv2.putText(frame, text, (x1 + 3, ty1 + th + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            except Exception as e:
                print(f"[WARN] pipeline error: {e}")
                cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 128, 128), 2)

        ok, buff = cv2.imencode('.jpg', frame)
        if not ok:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buff.tobytes() + b'\r\n')

    cap.release()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)