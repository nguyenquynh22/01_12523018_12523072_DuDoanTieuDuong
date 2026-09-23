# 07_12523018_12523072_DuDoanTieuDuong

## 1. Thành viên
| Họ và tên | MSSV | Phần việc đảm nhận |
| :--- | :--- | :--- |
| **Phan Tùng Dương** | 12523018 | Xây dựng Backend, cấu trúc Docker, huấn luyện model Logistic Regression & Naive Bayes, thiết lập Git & Merge |
| **Nguyễn Thị Như Quỳnh** | 12523072 | Xây dựng Frontend, phân tích EDA (≥5 hình), huấn luyện model SVM & Random Forest, viết báo cáo Word & Slide |

---

## 2. Bài toán
* **Mô tả bài toán:** Xây dựng hệ thống dự đoán nguy cơ mắc bệnh tiểu đường dựa trên các chỉ số y tế lâm sàng của bệnh nhân.
* **Loại bài toán:** Phân loại nhị phân (Binary Classification) (0: Bình thường, 1: Nguy cơ tiểu đường).
* **Cột mục tiêu (Target):** `Outcome`
* **Ý nghĩa thực tế:** Giúp sàng lọc sớm nguy cơ mắc bệnh tiểu đường cho người dân, hỗ trợ bác sĩ đưa ra quyết định chẩn đoán nhanh chóng và chính xác hơn.

---

## 3. Dữ liệu
* **Nguồn Kaggle:** [Pima Indians Diabetes Database](https://www.kaggle.com/datasets/mragpavank/diabetes) 
* **Giấy phép:** Phục vụ mục đích học tập và nghiên cứu (Open Database, CC0).
* **Mô tả các cột chính:** 
  * `Pregnancies`: Số lần mang thai
  * `Glucose`: Nồng độ Glucose trong máu
  * `BloodPressure`: Huyết áp tâm trương (mm Hg)
  * `SkinThickness`: Độ dày nếp gấp da cơ tam đầu (mm)
  * `Insulin`: Insulin huyết thanh 2 giờ (mu U/ml)
  * `BMI`: Chỉ số khối cơ thể (weight in kg / (height in m)²)
  * `DiabetesPedigreeFunction`: Hàm phả hệ tiểu đường
  * `Age`: Tuổi (năm)
* **Cách giải nén dataset.zip:** 
  Tải file `dataset.zip` từ thư mục `ai-models/data/` và giải nén trực tiếp tại chỗ để các script EDA đọc file CSV.

---

## 4. Kết quả model
*Bảng so sánh hiệu năng các mô hình trên tập Test:*

| Model | Accuracy | Precision | Recall | F1-Score | Thời gian dự đoán | Kích thước file | Nhận xét & Lựa chọn |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline** | 0.65 | 0.40 | 0.50 | 0.44 | 1 ms | - | Mức cơ sở ngẫu nhiên |
| **Logistic Regression** (Dương) | 0.77 | 0.72 | 0.58 | 0.64 | 2 ms | 15 KB | Ổn định, tuyến tính tốt |
| **Naive Bayes** (Dương) | 0.75 | 0.68 | 0.60 | 0.64 | 2 ms | 12 KB | Xử lý xác suất nhanh |
| **SVM** (Quỳnh) | 0.78 | 0.74 | 0.61 | 0.67 | 4 ms | 45 KB | Phân tách biên tốt |
| **Random Forest** (Quỳnh) | **0.81** | **0.78** | **0.68** | **0.72** | 8 ms | 1.2 MB | **Được chọn** do độ chính xác và F1-score cao nhất |

* **Model được chọn:** Random Forest.
* **Lý do:** Cho điểm F1-Score và Accuracy vượt trội hơn các mô hình còn lại, giảm thiểu tỷ lệ bỏ sót ca bệnh (False Negative) quan trọng trong y tế.

---

## 5. Đóng gói model
* **Đường dẫn file model trong repo:** `ai-models/models/model.joblib` (bao gồm cả Sklearn Pipeline tiền xử lý và mô hình Random Forest đã huấn luyện).
* **Cách export từ Colab:** 
  Sau khi chạy xong notebook `03_train.ipynb`, dùng thư viện `joblib` để nén cụm pipeline và model:
  ```python
  import joblib
  joblib.dump(pipeline_and_model, "model.joblib", compress=3)