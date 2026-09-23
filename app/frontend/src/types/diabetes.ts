// src/types/diabetes.ts

export interface DiabetesInput {
  pregnancies: number;
  glucose: number;
  bloodPressure: number;
  skinThickness: number;
  insulin: number;
  bmi: number;
  diabetesPedigreeFunction: number;
  age: number;
}

export interface ModelPredictionResult {
  modelName: string;         // Tên model: "Logistic Regression", "SVM", "Naive Bayes", "Random Forest"
  prediction: number;        // 0: Bình thường, 1: Nguy cơ tiểu đường
  probability: number;       // Xác suất % (Ví dụ: 85.5)
  statusText: string;        // "Nguy cơ cao" hoặc "An toàn"
  description: string;       // Mô tả ngắn về đặc trưng của model đó
}

export interface PredictionResponse {
  id: string;
  createdAt: string;
  inputData: DiabetesInput;
  results: ModelPredictionResult[]; // Mảng chứa kết quả của cả 4 model
  bestModel: string;                 // Tên model tốt nhất (VD: "Random Forest")
}