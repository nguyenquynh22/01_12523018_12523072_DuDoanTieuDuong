FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt ./

# Nâng cấp pip và thêm timeout để tránh lỗi rớt mạng khi tải các gói lớn
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --timeout=100 -r requirements.txt

COPY . .

EXPOSE 8001

CMD ["uvicorn", "service.main:app", "--host", "0.0.0.0", "--port", "8001"]