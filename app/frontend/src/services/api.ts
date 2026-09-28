import type { DiabetesInput, PredictionResponse } from "../types/diabetes";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export async function predictDiabetes(
  input: DiabetesInput,
): Promise<PredictionResponse> {
  const response = await fetch(`${API_URL}/api/v1/predict-disease`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(input),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(errorText || "Không thể gọi backend để dự đoán");
  }

  return (await response.json()) as PredictionResponse;
}
