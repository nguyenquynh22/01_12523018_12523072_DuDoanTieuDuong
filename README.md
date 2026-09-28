# Hệ Thống Dự Đoán Bệnh Tiểu Đường (Diabetes Prediction AI)

Bài tập lớn môn: Học máy cơ bản  
Bộ dữ liệu: Pima Indians Diabetes Database (Kaggle)  
Mô hình triển khai: Random Forest Pipeline (Độ chính xác: 81%, Macro F1: 72%)  
Kiến trúc: Microservices độc lập đóng gói hoàn chỉnh bằng Docker Compose

## Mục Lục

1. [Thành viên nhóm](#1-thành-viên-nhóm)
2. [Tổng quan dự án](#2-tổng-quan-dự-án)
3. [Kiến trúc hệ thống](#3-kiến-trúc-hệ-thống)
4. [Kết quả so sánh mô hình](#4-kết-quả-so-sánh-mô-hình)
5. [Hợp đồng API (API Endpoints)](#5-hợp-đồng-api-api-endpoints)
6. [Hướng dẫn vận hành hệ thống (Step-by-Step)](#6-hướng-dẫn-vận-hành-hệ-thống-step-by-step)
7. [Hướng dẫn test API từng bước](#7-hướng-dẫn-test-api-từng-bước)
8. [Spam test / Load test](#8-spam-test--load-test)
9. [Xem log](#9-xem-log)
10. [Xử lý sự cố](#10-xử-lý-sự-cố)
11. [Cấu trúc thư mục](#11-cấu-trúc-thư-mục)
12. [Demo online qua Ngrok](#12-demo-online-qua-ngrok)

---

## 1. Thành viên nhóm

| Họ và tên                |   MSSV   | Phần việc đảm nhận                                                                                                     |
| :----------------------- | :------: | :--------------------------------------------------------------------------------------------------------------------- |
| **Phan Tùng Dương**      | 12523018 | Xây dựng Backend, làm báo cáo word, huấn luyện model Logistic Regression & Naive Bayes                                 |
| **Nguyễn Thị Như Quỳnh** | 12523072 | Xây dựng Khung dự án, làm ai-service, frontend, làm slide báo cáo, phân tích EDA, huấn luyện model SVM & Random Forest |

---

## 2. Tổng Quan Dự Án

Dự án giải quyết bài toán phân loại nhị phân (Binary Classification) nhằm dự đoán nguy cơ mắc bệnh tiểu đường dựa trên các chỉ số y tế lâm sàng của bệnh nhân.  
Biến mục tiêu `Outcome`:

- **0:** Bình thường (Không mắc bệnh).
- **1:** Nguy cơ tiểu đường (Cần theo dõi và chẩn đoán thêm).

**Điểm nổi bật kỹ thuật:**

- **Chống rò rỉ dữ liệu (No Data Leakage):** Scaler chỉ fit duy nhất trên tập Train, biến đổi transform cho tập Test.
- **Đóng gói trọn gói:** Pipeline tiền xử lý và 4 mô hình được đóng gói chung trong `model.joblib`.
- **Khởi động tức thì (Instant Startup):** AI Service nạp model ngay khi container khởi động, không phát sinh độ trễ ở request đầu tiên.
- **Quan sát xuyên suốt:** Mọi request đều mang mã `X-Request-ID` được đồng bộ qua Frontend, Backend, AI Service và lưu vết vào MongoDB.

---

## 3. Kiến Trúc Hệ Thống

Toàn bộ hệ thống gồm các container độc lập kết nối qua mạng nội bộ Docker:

- **Frontend (Port 3000 / Nginx Gateway Port 8080):** Giao diện web hiện đại, hỗ trợ nhập liệu thông số y tế và hiển thị kết quả dự đoán trực quan.
- **Backend (Port 8000):** Node.js Express API Gateway xử lý logic, kiểm thực dữ liệu, chuyển tiếp sang AI Service và lưu lịch sử vào MongoDB.
- **AI Service (Port 8001):** FastAPI Microservice chuyên trách suy luận học máy từ các model đã huấn luyện.
- **MongoDB (Port 27017):** Lưu vết toàn bộ lịch sử dự đoán và thông số bệnh nhân.

---

## 4. Kết Quả So Sánh Mô Hình

### Bảng so sánh kết quả thực nghiệm trên tập Test

| Metric | Logistic Regression | Gaussian Naive Bayes | Support Vector Machine | Random Forest |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy (%)** | 70.78 | 70.13 | 74.03 | **77.92** |
| **Precision (%)** | 69.89 | 70.99 | 73.37 | **77.45** |
| **Recall (%)** | 70.78 | 70.13 | 74.03 | **77.92** |
| **F1-score (%)** | 70.08 | 70.45 | 73.49 | **77.32** |
| **ROC - AUC** | 0.8130 | 0.7646 | 0.7964 | **0.8192** |
| **Thời gian dự đoán (ms)** | **0.55** | 1.41 | 5.61 | 18.53 |
| **Nhận xét** | Tốc độ nhanh nhất, khả năng phân tách (ROC-AUC) tốt (0.8130). | Tốc độ rất nhanh, tuy nhiên các chỉ số đánh giá tổng quan thấp hơn các mô hình còn lại. | Cân bằng tốt giữa độ chính xác (74.03%) và thời gian xử lý (5.61 ms). | Hiệu suất tổng thể tối ưu nhất (Accuracy 77.92%, ROC-AUC cao nhất 0.8192). |
---

## 5. Hợp Đồng API (API Endpoints)

- **Endpoint:** `POST /api/predict` (qua Backend) hoặc thư viện gọi trực tiếp AI Service.
- **Request Body mẫu (`scripts/sample_request.json`):**
  ```json
  {
    "features": {
      "Pregnancies": 2,
      "Glucose": 120,
      "BloodPressure": 70,
      "SkinThickness": 20,
      "Insulin": 80,
      "BMI": 25.5,
      "DiabetesPedigreeFunction": 0.5,
      "Age": 30
    },
    "model_name": "random_forest"
  }
  ```
--- 

## 6. Hướng Dẫn Vận Hành Hệ Thống (Step-by-Step)

Phần này chia thành 2 cách tiếp cận tùy thuộc vào đối tượng sử dụng:

### 🟢 PHẦN A: Dành cho Người không biết gì về công nghệ (Chỉ xem và test giao diện nhanh)

* **Step 1:** Tải và cài đặt phần mềm [Docker Desktop](https://www.docker.com/products/docker-desktop/?utm_source=gemini), mở phần mềm lên và đợi đến khi biểu tượng góc trái chuyển sang màu xanh lá (**Running**).
* **Step 2:** Tải mã nguồn dự án về máy tính và giải nén thư mục (ví dụ thư mục `Diabetes`).
* **Step 3:** Mở thư mục dự án, click chuột phải chọn mở cửa sổ lệnh (Terminal / PowerShell / Command Prompt).
* **Step 4:** Chạy lệnh khởi động hệ thống:
```bash
docker compose up -d

````

- **Step 5:** Mở trình duyệt web và truy cập vào đường dẫn: **`http://localhost:8080`** để sử dụng giao diện trực quan. Khi dùng xong, tắt hệ thống bằng lệnh: `docker compose down`.

---

### 🔵 PHẦN B: Dành cho Người làm IT / Giám khảo kỹ thuật (Chạy Source Code & Test API)

- **Step 1 — Khởi động toàn bộ microservices:**

```bash
docker compose up --build -d
docker compose ps

```

- **Step 2 — Kiểm tra cổng dịch vụ nội bộ:**
- Giao diện Web / Nginx Gateway: `http://localhost:8080`
- API Backend (Swagger UI): `http://localhost:8000/docs`
- AI Service (FastAPI): `http://localhost:8001/docs`

- **Step 3 — Chạy kiểm thử tự động (Pytest):**

```bash
pytest tests/ -v

```

---

## 7. Hướng Dẫn Test API Từng Bước

_(Trên Windows PowerShell, hãy gõ `curl.exe` thay vì `curl`)._

- **Bước 1 — Kiểm tra server sống:**

```bash
curl.exe http://localhost:8000/health

```

- **Bước 2 — Gửi request dự đoán hợp lệ:**

```bash
curl.exe -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -H "X-Request-ID: test-001" \
  -d @scripts/sample_request.json

```

- **Bước 3 — Thử nghiệm dữ liệu lỗi (Kiểm tra bắt lỗi):**

```bash
curl.exe -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"Glucose": 99999}}'

```

- **Bước 4 — Xem lịch sử dự đoán từ MongoDB:**

```bash
curl.exe http://localhost:8000/api/history

```

---

## 8. Spam Test / Load Test

```bash
python scripts/load_test.py

```

---

## 9. Xem Log

```bash
docker compose logs -f                    # Xem toàn bộ log
docker compose logs -f backend            # Riêng log Backend
docker compose logs -f ai-service         # Riêng log AI Service

```

---

## 10. Xử Lý Sự Cố

- **Lỗi `port is already allocated`:** Cổng 3000/8000/8001/8080 đang bị chiếm. Hãy tắt ứng dụng đang dùng cổng đó hoặc đổi port trong file `docker-compose.yml`.
- **Lỗi kết nối AI Service / MongoDB:** Đảm bảo Docker Desktop đang chạy và dùng lệnh `docker compose restart` để khởi động lại các container.

---

## 11. Cấu Trúc Thư Mục

```text
├── app/
│   ├── frontend/                 # Giao diện web người dùng
│   └── backend/                  # Node.js API Gateway & MongoDB connection
├── ai-models/
│   ├── colab/                    # Notebook EDA, tiền xử lý, huấn luyện model
│   └── service/                  # FastAPI Microservice chứa model.joblib
├── nginx/                        # Cấu hình Nginx Gateway
├── docker-compose.yml
├── .env.example
└── README.md

```

---

## 12. Demo Online Qua Ngrok (Dùng để test từ xa)

Để mở truy cập công khai qua internet bằng domain tĩnh đã cấu hình của nhóm:

```bash
.\ngrok http --domain=unaltered-eagle-wackiness.ngrok-free.dev 8080

```
