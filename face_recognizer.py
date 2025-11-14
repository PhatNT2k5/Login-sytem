import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import pickle
import os
from pathlib import Path
from typing import Tuple

# Tắt thông báo không cần thiết
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class FaceRecognizer:
    """
    Class để nhận dạng khuôn mặt sử dụng mô hình YOLO và CSDL.
    Logic được trích từ 'webcam_face_id_use_yolo.py' - ĐÃ ĐỒNG BỘ HÓA.
    """
    
    def __init__(self, 
                 model_path="model_train/yolo-classification.pt", 
                 db_path="face_db.pkl", 
                 device="cpu", 
                 threshold=0.85):  # Tăng ngưỡng lên 0.85 để nghiêm ngặt hơn
        
        print(f"[INFO] Đang tải mô hình nhận dạng YOLO từ: {model_path}")
        self.device = torch.device(device)
        
        # Load model và tạo extractor giống webcam_face_id_use_yolo.py
        self.extractor, self.transform = self._prepare_model(model_path, self.device)
        
        print(f"[INFO] Đang tải CSDL khuôn mặt từ: {db_path}")
        self.db_path = db_path  # LƯU ĐỊA CHỈ DB
        self.db = self._load_db(db_path)
        self.threshold = threshold
        
        # Chuẩn bị person_data giống identify_loop
        self.person_data = self._prepare_person_data()

    def _get_transform(self):
        """Transform CHÍNH XÁC như webcam_face_id_use_yolo.py"""
        return transforms.Compose([
            transforms.Resize((224, 224)),  # Size 224, không phải 128
            transforms.ToTensor(),           # Chỉ scale về [0,1], KHÔNG normalize
        ])

    def _try_load_ultralytics(self, model_path):
        """Thử load bằng Ultralytics trước"""
        try:
            from ultralytics import YOLO
            return YOLO(model_path)
        except Exception:
            return None

    def _get_torch_module_from_ultra(self, ultra_model):
        """Lấy torch.nn.Module từ Ultralytics model"""
        m = getattr(ultra_model, 'model', None)
        inner = getattr(m, 'model', None) if m is not None else None
        if isinstance(inner, torch.nn.Module):
            return inner
        if isinstance(m, torch.nn.Module):
            return m
        return None

    def _load_checkpoint_as_module(self, path):
        """Load checkpoint trực tiếp - cho phép ClassificationModel"""
        try:
            from ultralytics.nn.tasks import ClassificationModel
            torch.serialization.add_safe_globals([ClassificationModel])
        except Exception:
            pass

        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        if isinstance(ckpt, torch.nn.Module):
            return ckpt
        if isinstance(ckpt, dict) and isinstance(ckpt.get('model', None), torch.nn.Module):
            return ckpt['model']
        raise RuntimeError("Checkpoint không phải torch.nn.Module")

    def _make_extractor(self, nn_module: torch.nn.Module):
        """
        Tạo extractor CHÍNH XÁC như webcam_face_id_use_yolo.py:
        - Tìm AdaptiveAvgPool2d cuối cùng
        - Hook vào đó để lấy embedding
        - Flatten và L2 normalize
        """
        import torch.nn as nn

        # Tìm AdaptiveAvgPool2d cuối cùng
        pool_mod = None
        for m in nn_module.modules():
            if isinstance(m, nn.AdaptiveAvgPool2d):
                pool_mod = m

        if pool_mod is None:
            kinds = {}
            for m in nn_module.modules():
                k = m.__class__.__name__
                kinds[k] = kinds.get(k, 0) + 1
            raise RuntimeError(
                "Không tìm thấy AdaptiveAvgPool2d trong mô hình. "
                f"Các lớp hiện có: {sorted((f'{k} x{kinds[k]}' for k in kinds))}"
            )

        store = {}

        def hook_fn(_, __, out):
            # out: (B, C, 1, 1) → (B, C)
            if isinstance(out, torch.Tensor):
                store['f'] = out.view(out.size(0), -1).detach()

        handle = pool_mod.register_forward_hook(hook_fn)

        def extractor(x: torch.Tensor):
            store.clear()
            with torch.no_grad():
                _ = nn_module(x)  # chạy forward để kích hoạt hook
            f = store.get('f')
            if f is None:
                raise RuntimeError("Không trích được embedding từ hook tại AdaptiveAvgPool2d.")
            return F.normalize(f, p=2, dim=1)  # (B, C) L2-normalize

        extractor._hook = handle
        return extractor

    def _prepare_model(self, model_path, device):
        """Chuẩn bị model GIỐNG HỆT webcam_face_id_use_yolo.py"""
        # Thử load bằng Ultralytics trước
        ultra = self._try_load_ultralytics(model_path)
        model = None
        
        if ultra is not None:
            m = self._get_torch_module_from_ultra(ultra)
            if m is not None:
                model = m.to(device).eval()
        
        # Fallback sang torch.load
        if model is None:
            model = self._load_checkpoint_as_module(model_path).to(device).eval()
        
        # Tạo extractor
        extractor = self._make_extractor(model)
        
        # Transform
        tf = self._get_transform()
        
        return extractor, tf

    def _load_db(self, path):
        """Load database"""
        if not os.path.exists(path):
            print(f"[WARN] Không tìm thấy file CSDL: {path}. Tạo CSDL rỗng.")
            return {}
        
        try:
            with open(path, 'rb') as f:
                db = pickle.load(f)
            return db
        except Exception as e:
            print(f"[ERROR] Lỗi khi tải CSDL {path}: {e}")
            return {}

    def _compute_centroid(self, embs):
        """Tính centroid từ list embeddings"""
        if not embs:
            return None
        X = np.stack(embs, 0)
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        c = X.mean(0)
        c = c / (np.linalg.norm(c) + 1e-12)
        return c.astype(np.float32)

    def _prepare_person_data(self):
        """Chuẩn bị person_data GIỐNG identify_loop"""
        person_data = {}
        for name, obj in self.db.items():
            embs = obj.get("embs", [])
            if not embs:
                continue
                
            # Tính centroid
            centroid = obj.get("centroid", None)
            if centroid is None:
                centroid = self._compute_centroid(embs)
            else:
                centroid = np.asarray(centroid, dtype=np.float32)
                
            if centroid is not None:
                person_data[name] = {
                    'centroid': centroid,
                    'embeddings': embs,
                    'num_samples': len(embs)
                }
        return person_data

    def _cosine(self, a, b):
        """Tính cosine similarity"""
        a = a / (np.linalg.norm(a) + 1e-12)
        b = b / (np.linalg.norm(b) + 1e-12)
        return float(np.dot(a, b))

    def _compute_similarity_stats(self, emb, embeddings_list):
        """Tính thống kê độ tương đồng"""
        if not embeddings_list:
            return 0.0, 0.0, 0.0
        
        similarities = [self._cosine(emb, stored_emb) for stored_emb in embeddings_list]
        max_sim = max(similarities)
        avg_sim = sum(similarities) / len(similarities)
        
        if len(similarities) > 1:
            std_sim = np.std(similarities)
        else:
            std_sim = 0.0
        
        return max_sim, avg_sim, std_sim

    def _adaptive_threshold(self, base_threshold, confidence_score, num_samples):
        """Tính threshold adaptive"""
        return max(base_threshold, 
                   base_threshold + (1.0 - confidence_score) * 0.1 + (1.0 / (num_samples + 1)) * 0.05)

    def _get_embedding(self, pil_image: Image.Image) -> np.ndarray:
        """
        Trích xuất embedding từ ảnh PIL.
        Trả về numpy array đã normalize.
        """
        # 1. Áp dụng transform
        img_t = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        # 2. Chạy extractor
        with torch.no_grad():
            emb = self.extractor(img_t)
        
        # 3. Chuyển về numpy
        return emb.squeeze(0).cpu().numpy().astype(np.float32)

    def recognize(self, pil_image: Image.Image) -> Tuple[str, float]:
        """
        Hàm chính: Nhận một ảnh PIL, trả về (name, score).
        Logic GIỐNG HỆT identify_loop trong webcam_face_id_use_yolo.py
        """
        try:
            if not self.person_data:
                return "Unknown", 0.0
            
            # 1. Lấy embedding
            emb = self._get_embedding(pil_image)
            
            # 2. So sánh với database (LOGIC GIỐNG identify_loop)
            best_name = "Unknown"
            best_score = -1.0
            best_confidence = 0.0
            
            for name, data in self.person_data.items():
                # Tính độ tương đồng với centroid
                centroid_sim = self._cosine(emb, data['centroid'])
                
                # Tính thống kê với tất cả embeddings
                max_sim, avg_sim, std_sim = self._compute_similarity_stats(emb, data['embeddings'])
                
                # Tính điểm tổng hợp (weighted score) - điều chỉnh trọng số
                combined_score = 0.6 * centroid_sim + 0.4 * max_sim  # Tăng trọng số cho centroid
                
                # Bonus nếu avg_sim cao
                if avg_sim > 0.5:  # Tăng ngưỡng avg_sim
                    combined_score += 0.15 * (avg_sim - 0.5)
                
                # Tính độ tin cậy
                confidence = 1.0 - min(std_sim, 0.25) / 0.25  # Giảm ngưỡng std để phát hiện biến thiên sớm hơn
                
                # Penalty mạnh hơn nếu std cao
                if std_sim > 0.15:  # Giảm ngưỡng std và tăng penalty
                    combined_score *= (1.0 - std_sim * 0.5)
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_name = name
                    best_confidence = confidence
            
            # 3. Điều chỉnh ngưỡng adaptive
            if best_name != "Unknown":
                num_samples = self.person_data[best_name]['num_samples']
                adaptive_thresh = self._adaptive_threshold(self.threshold, best_confidence, num_samples)
            else:
                adaptive_thresh = self.threshold
            
            # 4. Quyết định cuối cùng
            if best_score >= adaptive_thresh:
                return best_name, best_score
            else:
                return "Unknown", best_score
            
        except Exception as e:
            print(f"[ERROR] Lỗi khi nhận dạng: {e}")
            import traceback
            traceback.print_exc()
            return "Error", 0.0