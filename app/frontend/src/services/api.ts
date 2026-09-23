// src/services/api.ts
import type { DiabetesInput, PredictionResponse } from '../types/diabetes';

export async function predictDiabetes(input: DiabetesInput): Promise<PredictionResponse> {
  return new Promise((resolve) => {
    setTimeout(() => {
      resolve({
        id: Math.random().toString(36).substring(2, 9),
        createdAt: new Date().toLocaleTimeString() + ' - ' + new Date().toLocaleDateString(),
        inputData: input,
        results: [
          {
            modelName: "Logistic Regression",
            prediction: input.glucose > 140 ? 1 : 0,
            probability: 76.4,
            statusText: input.glucose > 140 ? "Nguy cơ tiểu đường" : "Bình thường",
            description: "Mô hình tuyến tính cơ sở, ổn định."
          },
          {
            modelName: "Support Vector Machine (SVM)",
            prediction: input.bmi > 30 ? 1 : 0,
            probability: 81.2,
            statusText: input.bmi > 30 ? "Nguy cơ tiểu đường" : "Bình thường",
            description: "Tối ưu hóa siêu phẳng phân tách lề."
          },
          {
            modelName: "Naive Bayes",
            prediction: input.glucose > 135 ? 1 : 0,
            probability: 74.5,
            statusText: input.glucose > 135 ? "Nguy cơ tiểu đường" : "Bình thường",
            description: "Dựa trên xác suất thống kê các biến độc lập."
          },
          {
            modelName: "Random Forest (Ensemble)",
            prediction: (input.glucose > 130 && input.bmi > 28) ? 1 : 0,
            probability: 89.8,
            statusText: (input.glucose > 130 && input.bmi > 28) ? "Nguy cơ tiểu đường" : "Bình thường",
            description: "Mô hình Bagging mạnh nhất, xử lý nhiễu tốt."
          }
        ],
        bestModel: "Random Forest (Ensemble)"
      });
    }, 500);
  });
}