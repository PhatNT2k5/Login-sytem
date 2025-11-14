"""
Module quản lý MongoDB cho lưu trữ vector đặc trưng khuôn mặt
"""
from pymongo import MongoClient
from datetime import datetime
import numpy as np
import json

class MongoDBManager:
    def __init__(self, uri="mongodb://localhost:27017/", db_name="face_recognition"):
        """
        Kết nối tới MongoDB
        
        Args:
            uri: MongoDB connection string
            db_name: Tên database
        """
        try:
            self.client = MongoClient(uri)
            self.db = self.client[db_name]
            self.users_collection = self.db["users"]
            self.attendance_collection = self.db["attendance"]
            
            # Tạo index cho tìm kiếm nhanh
            self.users_collection.create_index("username", unique=True)
            self.attendance_collection.create_index("username")
            self.attendance_collection.create_index("timestamp")
            
            print(f"[INFO] Đã kết nối tới MongoDB database: {db_name}")
            self.is_connected = True
        except Exception as e:
            print(f"[ERROR] Không thể kết nối MongoDB: {e}")
            self.client = None
            self.db = None
            self.is_connected = False
    
    def save_user_embeddings(self, username, embeddings, metadata=None):
        """
        Lưu vector đặc trưng của một user
        
        Args:
            username: Tên người dùng
            embeddings: List các numpy arrays (vectors)
            metadata: Dict thêm thông tin (email, phone, etc.)
        
        Returns:
            bool: True nếu lưu thành công
        """
        if self.db is None:
            print("[ERROR] Không có kết nối MongoDB")
            return False
        
        try:
            # Chuyển numpy arrays thành lists để có thể serialize
            embeddings_lists = [emb.tolist() if isinstance(emb, np.ndarray) else emb 
                               for emb in embeddings]
            
            # Tính centroid
            centroid = np.mean(embeddings, axis=0).tolist()
            
            # Tạo document
            user_doc = {
                "username": username,
                "embeddings": embeddings_lists,
                "centroid": centroid,
                "created_at": datetime.now(),
                "updated_at": datetime.now(),
                "sample_count": len(embeddings_lists),
                "metadata": metadata or {}
            }
            
            # Lưu hoặc cập nhật
            result = self.users_collection.update_one(
                {"username": username},
                {"$set": user_doc},
                upsert=True
            )
            
            if result.upserted_id or result.modified_count > 0:
                print(f"[SUCCESS] Đã lưu {len(embeddings_lists)} vector cho user '{username}'")
                return True
            else:
                print(f"[WARN] Không thay đổi gì cho user '{username}'")
                return False
                
        except Exception as e:
            print(f"[ERROR] Lỗi khi lưu embeddings: {e}")
            return False
    
    def get_user_embeddings(self, username):
        """
        Lấy vector đặc trưng của một user
        
        Args:
            username: Tên người dùng
        
        Returns:
            dict hoặc None: Document user hoặc None nếu không tìm thấy
        """
        if self.db is None:
            return None
        
        try:
            user_doc = self.users_collection.find_one({"username": username})
            return user_doc
        except Exception as e:
            print(f"[ERROR] Lỗi khi lấy embeddings: {e}")
            return None
    
    def user_exists(self, username):
        """Kiểm tra user có tồn tại hay không"""
        if self.db is None:
            return False
        
        try:
            return self.users_collection.find_one({"username": username}) is not None
        except Exception as e:
            print(f"[ERROR] Lỗi khi kiểm tra user: {e}")
            return False
    
    def get_all_users(self):
        """Lấy danh sách tất cả users"""
        if self.db is None:
            return []
        
        try:
            users = list(self.users_collection.find({}, {"_id": 0}))
            return users
        except Exception as e:
            print(f"[ERROR] Lỗi khi lấy danh sách users: {e}")
            return []
    
    def log_attendance(self, username, action, timestamp=None):
        """
        Ghi log chấm công
        
        Args:
            username: Tên người dùng
            action: "IN" hoặc "OUT"
            timestamp: Thời gian (mặc định là hiện tại)
        """
        if self.db is None:
            return False
        
        try:
            attendance_doc = {
                "username": username,
                "action": action,
                "timestamp": timestamp or datetime.now()
            }
            
            result = self.attendance_collection.insert_one(attendance_doc)
            return result.inserted_id is not None
        except Exception as e:
            print(f"[ERROR] Lỗi khi ghi log chấm công: {e}")
            return False
    
    def get_attendance_history(self, username, limit=50):
        """Lấy lịch sử chấm công của user"""
        if self.db is None:
            return []
        
        try:
            history = list(self.attendance_collection.find(
                {"username": username},
                {"_id": 0}
            ).sort("timestamp", -1).limit(limit))
            return history
        except Exception as e:
            print(f"[ERROR] Lỗi khi lấy lịch sử: {e}")
            return []
    
    def delete_user(self, username):
        """Xóa user khỏi database"""
        if self.db is None:
            return False
        
        try:
            result = self.users_collection.delete_one({"username": username})
            return result.deleted_count > 0
        except Exception as e:
            print(f"[ERROR] Lỗi khi xóa user: {e}")
            return False
    
    def close(self):
        """Đóng kết nối MongoDB"""
        if self.client:
            self.client.close()
            print("[INFO] Đã đóng kết nối MongoDB")
