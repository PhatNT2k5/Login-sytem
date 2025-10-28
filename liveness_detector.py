import cv2
import numpy as np
import onnxruntime as ort
import os
from typing import Tuple

def _softmax(x: np.ndarray) -> np.ndarray:
    # vì sao: UI đang so ngưỡng trên logit -> cần softmax để ra xác suất
    x = x.astype(np.float32)
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-12)

class LivenessFAS:
    """
    ONNX Face Anti-Spoofing.
    """
    def __init__(
        self,
        model_path: str = "liveness_fas.onnx",
        input_size: Tuple[int, int] = (224, 224),
        mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        std: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        preprocess_mode: str = "div255_only",  # ["div255_only", "mean_std"]
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Không tìm thấy file model Liveness: {model_path}")

        so = ort.SessionOptions()
        so.log_severity_level = 3
        providers = ['CPUExecutionProvider']
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            providers.insert(0, 'CUDAExecutionProvider')

        self.session = ort.InferenceSession(model_path, so, providers=providers)
        input_details = self.session.get_inputs()[0]
        self.input_name = input_details.name
        self.input_shape = input_details.shape

        # Suy ra kích thước model
        if len(self.input_shape) == 4:
            if self.input_shape[-1] == 3:
                model_h, model_w = self.input_shape[1], self.input_shape[2]
            else:
                model_h, model_w = self.input_shape[2], self.input_shape[3]
            try:
                model_h, model_w = int(model_h), int(model_w)
                self.input_size = (model_h, model_w)
            except (ValueError, TypeError):
                self.input_size = input_size
        else:
            self.input_size = input_size

        self.mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(std, dtype=np.float32).reshape(1, 1, 3)
        self.labels = ["SPOOF", "REAL"]
        self.preprocess_mode = preprocess_mode  # vì sao: cho phép khớp đúng data-prep khi train

    def _preprocess(self, image_bgr: np.ndarray) -> np.ndarray:
        img = cv2.resize(image_bgr, self.input_size, interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        if self.preprocess_mode == "mean_std":
            img = (img - self.mean) / self.std  # [-1,1] khi mean=std=0.5
        # div255_only: giữ nguyên [0,1] như nhiều repo FAS dùng

        return np.expand_dims(img, axis=0)  # NHWC

    def predict(self, face_crop_bgr: np.ndarray) -> Tuple[str, float]:
        """
        Trả về (label, prob_real). prob_real là xác suất (đã softmax) của lớp REAL.
        """
        x = self._preprocess(face_crop_bgr)
        outputs = self.session.run(None, {self.input_name: x})
        scores = outputs[0][0]

        # Chuẩn hoá thành prob
        if scores.ndim == 1 and scores.shape[0] == 2:
            probs = _softmax(scores)  # [p_spoof, p_real]
            prob_real = float(probs[1])
            label = "REAL" if prob_real >= 0.5 else "SPOOF"
        else:
            # single-logit realness → sigmoid
            s = float(scores[0])
            prob_real = 1.0 / (1.0 + np.exp(-s))
            label = "REAL" if prob_real >= 0.5 else "SPOOF"

        return label, prob_real