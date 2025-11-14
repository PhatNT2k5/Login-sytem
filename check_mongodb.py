"""
Script kiểm tra kết nối và dữ liệu MongoDB
Chạy: python check_mongodb.py
"""

from mongo_db import MongoDBManager
import json

def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)

def main():
    print_section("KIỂM TRA MONGODB")
    
    # Kết nối
    print("\n[1] Kết nối tới MongoDB...")
    mongo_db = MongoDBManager(uri="mongodb://localhost:27017/", db_name="face_recognition")
    
    if not mongo_db.is_connected:
        print("❌ Lỗi: Không thể kết nối MongoDB!")
        print("   - Kiểm tra MongoDB có đang chạy trên localhost:27017?")
        print("   - Chạy: mongod")
        return
    
    print("✅ Kết nối thành công!")
    
    # Lấy danh sách users
    print("\n[2] Lấy danh sách users...")
    users = mongo_db.get_all_users()
    print(f"   Tổng users: {len(users)}")
    
    if users:
        print_section("DANH SÁCH USERS")
        for i, user in enumerate(users, 1):
            emb_count = len(user.get("embeddings", []))
            print(f"\n{i}. {user.get('username')}")
            print(f"   - Samples: {user.get('sample_count', 0)}")
            print(f"   - Created: {user.get('created_at')}")
            print(f"   - Status: {user.get('metadata', {}).get('status', 'unknown')}")
    
    # Lấy chi tiết user nếu có
    if users:
        username = users[0].get('username')
        print_section(f"CHI TIẾT USER: {username}")
        
        user_doc = mongo_db.get_user_embeddings(username)
        print(f"\nTên: {user_doc.get('username')}")
        print(f"Số mẫu: {user_doc.get('sample_count')}")
        print(f"Centroid shape: {len(user_doc.get('centroid', []))} features")
        print(f"Tạo ngày: {user_doc.get('created_at')}")
        print(f"Cập nhật: {user_doc.get('updated_at')}")
        
        # Lịch sử chấm công
        history = mongo_db.get_attendance_history(username, limit=10)
        if history:
            print(f"\n📋 Lịch sử chấm công ({len(history)} bản ghi):")
            for record in history:
                print(f"   - {record.get('action')}: {record.get('timestamp')}")
    
    # Thống kê
    print_section("THỐNG KÊ")
    total_samples = sum(u.get('sample_count', 0) for u in users)
    print(f"\nTổng users: {len(users)}")
    print(f"Tổng samples: {total_samples}")
    if len(users) > 0:
        print(f"Trung bình samples/user: {total_samples / len(users):.1f}")
    
    mongo_db.close()
    print_section("HOÀN THÀNH")
    print("\n✅ Kiểm tra hoàn tất!")

if __name__ == "__main__":
    main()
