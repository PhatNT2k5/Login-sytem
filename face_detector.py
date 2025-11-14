import cv2
import os
import numpy as np

class FaceDetectorYuNet:
    """
    YuNet detector (OpenCV). Quan trọng: setInputSize theo đúng (W,H) của frame mỗi lần.
    """
    def __init__(self, model_path="face_detection_yunet_2023mar.onnx",
                 conf_threshold=0.8, nms_threshold=0.3, top_k=1,
                 init_input_size=(640, 480)):
        # Tạo detector; input size lúc khởi tạo chỉ là giá trị ban đầu
        if os.path.exists(model_path):
            print(f"[INFO] Load YuNet: {model_path}")
            self.detector = cv2.FaceDetectorYN.create(
                model_path, "", init_input_size, conf_threshold, nms_threshold, top_k
            )
        else:
            # Nếu OpenCV của bạn có model built-in
            print(f"[WARN] Không thấy {model_path}. Thử YuNet mặc định của OpenCV.")
            self.detector = cv2.FaceDetectorYN.create(
                "", "", init_input_size, conf_threshold, nms_threshold, top_k
            )

    def detect(self, image_bgr):
        # Bảo đảm ảnh là BGR 3 kênh
        if len(image_bgr.shape) == 2 or (len(image_bgr.shape) == 3 and image_bgr.shape[2] == 1):
            image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)

        H, W = image_bgr.shape[:2]
        # *** QUAN TRỌNG: setInputSize THEO (W,H) CỦA ẢNH HIỆN TẠI ***
        self.detector.setInputSize((W, H))

        _, faces = self.detector.detect(image_bgr)
        if faces is None:
            return []

        # Trả box dạng [x, y, w, h]
        face_boxes = []
        for f in faces:
            x, y, w, h = f[:4].astype(int)
            # lọc box ngoài ảnh
            if w <= 0 or h <= 0:
                continue
            # clamp
            x = max(0, min(x, W - 1))
            y = max(0, min(y, H - 1))
            w = min(w, W - x)
            h = min(h, H - y)
            if w > 0 and h > 0:
                face_boxes.append([x, y, w, h])
        return face_boxes


class FaceDetectorHaar:
    """
    Haar cascade fallback.
    """
    def __init__(self, scale_factor=1.1, min_neighbors=5, min_size=(100, 100)):
        haar_xml_file = 'haarcascade_frontalface_default.xml'
        haar_model_path = os.path.join(cv2.data.haarcascades, haar_xml_file)
        if not os.path.exists(haar_model_path):
            raise FileNotFoundError(f"Thiếu Haar XML: {haar_model_path}")
        print(f"[INFO] Load Haar: {haar_model_path}")
        self.detector = cv2.CascadeClassifier(haar_model_path)
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size

    def detect(self, gray_image):
        faces = self.detector.detectMultiScale(
            gray_image,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size,
        )
        return faces