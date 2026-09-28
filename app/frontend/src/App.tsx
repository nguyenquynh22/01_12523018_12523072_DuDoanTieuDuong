// src/App.tsx
import { useState } from "react";
import { PredictionForm } from "./components/PredictionForm";
import { ResultComparison } from "./components/ResultComparison";
import { predictDiabetes } from "./services/api";
import type { DiabetesInput, PredictionResponse } from "./types/diabetes";

export default function App() {
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [latestResult, setLatestResult] = useState<PredictionResponse | null>(
    null,
  );

  const handlePredict = async (inputData: DiabetesInput) => {
    setIsLoading(true);
    try {
      const response = await predictDiabetes(inputData);
      setLatestResult(response);
    } catch (error) {
      console.error("Lỗi khi gọi API dự đoán:", error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app-shell">
      <main className="page-container">
        <header className="page-header">
          <div className="eyebrow">AI HEALTH CHECK</div>
          <h1>Dự đoán nguy cơ tiểu đường</h1>
          <p>
            Chỉ cần nhập các chỉ số y tế, hệ thống sẽ chạy đồng thời 4 mô hình
            AI và đưa ra kết quả tốt nhất.
          </p>
        </header>

        <section className="panel-grid">
          <PredictionForm onSubmit={handlePredict} isLoading={isLoading} />

          {latestResult ? (
            <ResultComparison data={latestResult} />
          ) : (
            <div className="empty-state">
              <p>
                Hãy nhập dữ liệu và nhấn <strong>Chạy Dự Đoán 4 Model</strong>
              </p>
              <span>Kết quả so sánh 4 mô hình sẽ xuất hiện ở đây.</span>
            </div>
          )}
        </section>
      </main>
    </div>
  );
}
