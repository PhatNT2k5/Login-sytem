import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify, request
from PIL import Image

from face_detector import FaceDetectorYuNet, FaceDetectorHaar
from liveness_detector import LivenessFAS
from face_recognizer import FaceRecognizer
from mongo_db import MongoDBManager
from datetime import datetime
import json
import os
import base64
import pickle
from pathlib import Path

YUNET_MODEL = "model/face_detection_yunet_2023mar.onnx"
LIVENESS_MODEL = "model/liveness_fas_in224.onnx"
YOLO_CLS_MODEL = "model/yolo-classification.pt"
FACE_DB = "model/face_db.pkl"

ATTENDANCE_LOG = "attendance_log.txt"
ATTENDANCE_STATE = "attendance_state.json"

current_detection = {
    "name": "Unknown",
    "is_live": False,
    "confidence": 0.0,
    "liveness_score": 0.0
}

# State cho chế độ đăng ký
register_state = {
    "active": False,
    "user_name": None,
    "samples_count": 0,
    "max_samples": 15,
    "embeddings": []
}

# Lưu face crop mới nhất cho đăng ký
latest_face_crop = None

LIVE_MIN_SCORE = 0.495  # ngưỡng trên prob_real

app = Flask(__name__)

# Khởi tạo MongoDB Manager
# Sửa URI nếu cần (nếu MongoDB chạy trên máy khác hoặc có mật khẩu)
try:
    mongo_db = MongoDBManager(uri="mongodb://localhost:27017/", db_name="face_recognition")
    MONGODB_AVAILABLE = mongo_db.is_connected
except Exception as e:
    print(f"[WARN] MongoDB không khả dụng: {e}")
    mongo_db = None
    MONGODB_AVAILABLE = False

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
        threshold=0.75,
    )
except Exception as e:
    print(f"[ERROR] Không thể tải mô hình: {e}")
    raise

def _expand_box(x, y, w, h, W, H, margin_ratio=0.15):
    dw = int(w * margin_ratio)
    dh = int(h * margin_ratio)
    x1 = max(0, x - dw); y1 = max(0, y - dh)
    x2 = min(W, x + w + dw); y2 = min(H, y + h + dh)
    return x1, y1, x2, y2

def load_attendance_state():
    """Load trạng thái IN/OUT của từng user"""
    if os.path.exists(ATTENDANCE_STATE):
        with open(ATTENDANCE_STATE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_attendance_state(state):
    """Lưu trạng thái IN/OUT"""
    with open(ATTENDANCE_STATE, 'w', encoding='utf-8') as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

def log_attendance(name, action):
    """Ghi log chấm công"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] - {name} - {action}\n"
    
    with open(ATTENDANCE_LOG, 'a', encoding='utf-8') as f:
        f.write(log_entry)
    
    return timestamp


def generate_frames():
    global latest_face_crop
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

            # LƯU face_crop MỚI NHẤT CHO ĐĂNG KÝ
            globals()['latest_face_crop'] = face_crop

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
                    # Cập nhật trạng thái
                    current_detection.update({
                        "name": "Unknown",
                        "is_live": False,
                        "confidence": 0.0,
                        "liveness_score": prob_real
                    })
                else:
                    pil_img = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
                    name, id_score = face_recognizer.recognize(pil_img)
                    disp_name = name if id_score >= face_recognizer.threshold else "Unknown"
                    info_label = f"{disp_name} ({id_score:.2f})"
                    box_color = (0, 255, 0)
                    
                    # Cập nhật trạng thái - CHỈ KHI NHẬN DIỆN THÀNH CÔNG
                    current_detection.update({
                        "name": disp_name,
                        "is_live": True,
                        "confidence": id_score,
                        "liveness_score": prob_real
                    })

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

# ===== ĐĂNG KÝ KHUÔN MẶT =====

@app.route("/register_start", methods=["POST"])
def register_start():
    """Bắt đầu quá trình đăng ký cho một người dùng"""
    try:
        data = request.get_json()
        user_name = data.get("name", "").strip()
        
        if not user_name:
            return jsonify({"success": False, "message": "Vui lòng nhập tên người dùng"}), 400
        
        # Kiểm tra user đã tồn tại
        if user_name in face_recognizer.db:
            return jsonify({"success": False, "message": f"Người dùng '{user_name}' đã tồn tại"}), 400
        
        # Khởi tạo state đăng ký
        register_state["active"] = True
        register_state["user_name"] = user_name
        register_state["samples_count"] = 0
        register_state["embeddings"] = []
        
        return jsonify({
            "success": True,
            "message": f"Bắt đầu đăng ký cho '{user_name}'. Vui lòng đưa khuôn mặt vào camera",
            "max_samples": register_state["max_samples"]
        })
    except Exception as e:
        print(f"[ERROR] register_start: {e}")
        return jsonify({"success": False, "message": str(e)}), 500


@app.route("/register_capture", methods=["POST"])
def register_capture():
    """Capture một mẫu khuôn mặt trong quá trình đăng ký"""
    global latest_face_crop
    try:
        if not register_state["active"]:
            print("[DEBUG] register_state not active")
            return jsonify({"success": False, "message": "Không có phiên đăng ký nào đang hoạt động"}), 400
        
        # Kiểm tra nếu đã có đủ mẫu
        if register_state["samples_count"] >= register_state["max_samples"]:
            print(f"[DEBUG] Already have {register_state['samples_count']} samples")
            return jsonify({
                "success": False, 
                "message": f"Đã lưu đủ {register_state['max_samples']} mẫu. Vui lòng hoàn tất đăng ký"
            }), 400
        
        # Kiểm tra liveness (không cần kiểm tra name vì đang đăng ký mới)
        if not current_detection["is_live"]:
            print(f"[DEBUG] Not live. is_live={current_detection['is_live']}")
            return jsonify({
                "success": False, 
                "message": "Không phải người thật. Vui lòng kiểm tra lại"
            }), 400
        
        # Lấy embedding từ khuôn mặt hiện tại
        if latest_face_crop is None:
            print("[DEBUG] latest_face_crop is None")
            return jsonify({
                "success": False,
                "message": "Chưa có dữ liệu khuôn mặt. Vui lòng thử lại"
            }), 400
        
        try:
            print(f"[DEBUG] Capturing sample. latest_face_crop shape: {latest_face_crop.shape}")
            face_crop = latest_face_crop.copy()
            pil_img = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
            emb = face_recognizer._get_embedding(pil_img)
            
            # Lưu embedding
            register_state["embeddings"].append(emb)
            register_state["samples_count"] += 1
            
            print(f"[SUCCESS] Sample #{register_state['samples_count']} saved")
            return jsonify({
                "success": True,
                "message": f"Lưu mẫu #{register_state['samples_count']} thành công",
                "samples_count": register_state["samples_count"],
                "max_samples": register_state["max_samples"]
            })
        except Exception as e:
            print(f"[ERROR] Lỗi khi tạo embedding: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({
                "success": False,
                "message": f"Lỗi khi xử lý khuôn mặt: {e}"
            }), 500
    
    except Exception as e:
        print(f"[ERROR] register_capture exception: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "message": str(e)}), 500


@app.route("/register_complete", methods=["POST"])
def register_complete():
    """Hoàn tất quá trình đăng ký"""
    try:
        if not register_state["active"]:
            return jsonify({"success": False, "message": "Không có phiên đăng ký nào đang hoạt động"}), 400
        
        if register_state["samples_count"] < 3:
            return jsonify({
                "success": False,
                "message": f"Cần ít nhất 3 mẫu (hiện có {register_state['samples_count']})"
            }), 400
        
        user_name = register_state["user_name"]
        embeddings = register_state["embeddings"]
        
        # Tính centroid
        centroid = face_recognizer._compute_centroid(embeddings)
        
        # Lưu vào database
        face_recognizer.db[user_name] = {
            "embs": embeddings,
            "centroid": centroid
        }
        
        # Lưu file DB
        Path(face_recognizer.db_path if hasattr(face_recognizer, 'db_path') else FACE_DB).parent.mkdir(parents=True, exist_ok=True)
        db_path = face_recognizer.db_path if hasattr(face_recognizer, 'db_path') else FACE_DB
        with open(db_path, "wb") as f:
            pickle.dump(face_recognizer.db, f)
        
        # ===== LƯU VÀO MONGODB =====
        if MONGODB_AVAILABLE and mongo_db:
            try:
                # Chuyển embeddings (numpy arrays) thành lists
                embeddings_for_mongo = [emb.tolist() if isinstance(emb, np.ndarray) else emb 
                                       for emb in embeddings]
                
                # Lưu vào MongoDB
                success = mongo_db.save_user_embeddings(
                    username=user_name,
                    embeddings=embeddings,  # Truyền numpy arrays, MongoDBManager sẽ chuyển đổi
                    metadata={
                        "registration_date": datetime.now().isoformat(),
                        "sample_count": len(embeddings),
                        "status": "active"
                    }
                )
                
                if success:
                    print(f"[SUCCESS] Đã lưu vector cho '{user_name}' vào MongoDB")
                else:
                    print(f"[WARN] Lưu MongoDB thất bại nhưng đã lưu vào file")
            except Exception as e:
                print(f"[WARN] Lỗi khi lưu MongoDB (nhưng vẫn lưu file): {e}")
        # ===========================
        
        # Cập nhật person_data
        face_recognizer.person_data = face_recognizer._prepare_person_data()
        
        # Reset state
        register_state["active"] = False
        register_state["user_name"] = None
        register_state["samples_count"] = 0
        register_state["embeddings"] = []
        
        return jsonify({
            "success": True,
            "message": f"✅ Đã đăng ký thành công '{user_name}' với {len(embeddings)} mẫu",
            "mongodb_saved": MONGODB_AVAILABLE
        })
    
    except Exception as e:
        print(f"[ERROR] register_complete: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "message": str(e)}), 500


@app.route("/register_cancel", methods=["POST"])
def register_cancel():
    """Hủy quá trình đăng ký"""
    register_state["active"] = False
    register_state["user_name"] = None
    register_state["samples_count"] = 0
    register_state["embeddings"] = []
    
    return jsonify({"success": True, "message": "Đã hủy phiên đăng ký"})


@app.route("/get_register_status", methods=["GET"])
def get_register_status():
    """Lấy trạng thái hiện tại của phiên đăng ký"""
    return jsonify({
        "active": register_state["active"],
        "user_name": register_state["user_name"],
        "samples_count": register_state["samples_count"],
        "max_samples": register_state["max_samples"]
    })

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


# ===== API QUẢN LÝ MONGODB =====

@app.route("/api/users", methods=["GET"])
def get_all_users():
    """Lấy danh sách tất cả users đã đăng ký"""
    if not MONGODB_AVAILABLE or not mongo_db:
        return jsonify({"success": False, "message": "MongoDB không khả dụng"}), 500
    
    try:
        users = mongo_db.get_all_users()
        # Loại bỏ embeddings khỏi response (quá lớn)
        for user in users:
            if "embeddings" in user:
                user["embeddings"] = [f"vector_{i}" for i in range(len(user["embeddings"]))]
        
        return jsonify({
            "success": True,
            "total": len(users),
            "users": users
        })
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@app.route("/api/user/<username>", methods=["GET"])
def get_user_info(username):
    """Lấy thông tin chi tiết của một user (không gồm embeddings)"""
    if not MONGODB_AVAILABLE or not mongo_db:
        return jsonify({"success": False, "message": "MongoDB không khả dụng"}), 500
    
    try:
        user_doc = mongo_db.get_user_embeddings(username)
        if not user_doc:
            return jsonify({"success": False, "message": f"User '{username}' không tồn tại"}), 404
        
        # Loại bỏ embeddings chi tiết
        user_info = {k: v for k, v in user_doc.items() if k != "embeddings"}
        user_info["embeddings_count"] = len(user_doc.get("embeddings", []))
        
        return jsonify({
            "success": True,
            "user": user_info
        })
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@app.route("/api/attendance/<username>", methods=["GET"])
def get_user_attendance(username):
    """Lấy lịch sử chấm công của user"""
    if not MONGODB_AVAILABLE or not mongo_db:
        return jsonify({"success": False, "message": "MongoDB không khả dụng"}), 500
    
    try:
        limit = request.args.get("limit", 50, type=int)
        history = mongo_db.get_attendance_history(username, limit=limit)
        
        return jsonify({
            "success": True,
            "username": username,
            "total": len(history),
            "history": history
        })
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@app.route("/api/user/<username>", methods=["DELETE"])
def delete_user(username):
    """Xóa user khỏi database"""
    if not MONGODB_AVAILABLE or not mongo_db:
        return jsonify({"success": False, "message": "MongoDB không khả dụng"}), 500
    
    try:
        success = mongo_db.delete_user(username)
        if success:
            return jsonify({"success": True, "message": f"Đã xóa user '{username}'"})
        else:
            return jsonify({"success": False, "message": f"User '{username}' không tồn tại"}), 404
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@app.route("/api/stats", methods=["GET"])
def get_stats():
    """Thống kê chung"""
    if not MONGODB_AVAILABLE or not mongo_db:
        return jsonify({"success": False, "message": "MongoDB không khả dụng"}), 500
    
    try:
        all_users = mongo_db.get_all_users()
        total_users = len(all_users)
        total_samples = sum(u.get("sample_count", 0) for u in all_users)
        
        return jsonify({
            "success": True,
            "total_users": total_users,
            "total_samples": total_samples,
            "average_samples_per_user": round(total_samples / total_users, 2) if total_users > 0 else 0
        })
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

# ===============================



@app.route("/get_status")
def get_status():
    """API trả về trạng thái nhận diện hiện tại"""
    state = load_attendance_state()
    user_state = state.get(current_detection["name"], "OUT")
    
    return jsonify({
        "name": current_detection["name"],
        "is_live": current_detection["is_live"],
        "confidence": current_detection["confidence"],
        "liveness_score": current_detection["liveness_score"],
        "current_state": user_state,
        "ready": current_detection["is_live"] and current_detection["name"] != "Unknown"
    })

@app.route("/checkin", methods=["POST"])
def checkin():
    """Xử lý chấm công"""
    if current_detection["name"] == "Unknown" or not current_detection["is_live"]:
        return jsonify({
            "success": False,
            "message": "Không nhận diện được người dùng hoặc không phải người thật"
        })
    
    name = current_detection["name"]
    state = load_attendance_state()
    
    # Xác định IN hay OUT
    current_state = state.get(name, "OUT")
    new_action = "IN" if current_state == "OUT" else "OUT"
    
    # Ghi log
    timestamp = log_attendance(name, new_action)
    
    # Cập nhật trạng thái
    state[name] = new_action
    save_attendance_state(state)
    
    # ===== LƯU VÀO MONGODB =====
    if MONGODB_AVAILABLE and mongo_db:
        try:
            mongo_db.log_attendance(name, new_action, timestamp)
            print(f"[SUCCESS] Đã ghi log '{name} - {new_action}' vào MongoDB")
        except Exception as e:
            print(f"[WARN] Lỗi khi ghi log MongoDB: {e}")
    # ===========================
    
    return jsonify({
        "success": True,
        "name": name,
        "action": new_action,
        "timestamp": timestamp,
        "message": f"Chấm công thành công: {name} - {new_action}"
    })


if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=5000, debug=False)
    finally:
        # Đóng kết nối MongoDB khi tắt app
        if mongo_db:
            mongo_db.close()