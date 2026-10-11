import { useState } from "react";
import { PredictionForm } from "./components/PredictionForm";
import { ResultComparison } from "./components/ResultComparison";
import { predictDiabetes } from "./services/api";
import type { DiabetesInput, PredictionResponse } from "./types/diabetes";

export default function App() {
  const [isLoading, setIsLoading] = useState(false);
  const [latestResult, setLatestResult] = useState<PredictionResponse | null>(
    null,
  );
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const handlePredict = async (inputData: DiabetesInput) => {
    setIsLoading(true);
    setErrorMessage(null);
    try {
      setLatestResult(await predictDiabetes(inputData));
    } catch (error) {
      const message =
        error instanceof Error
          ? error.message
          : "Không thể kết nối tới backend để dự đoán.";
      setErrorMessage(message);
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
        </header>
        <section className="panel-grid">
          <PredictionForm onSubmit={handlePredict} isLoading={isLoading} />
          {errorMessage ? (
            <div className="empty-state border border-red-200 bg-red-50 text-red-700">
              <p className="font-semibold">Lỗi kết nối API</p>
              <span>{errorMessage}</span>
            </div>
          ) : latestResult ? (
            <ResultComparison data={latestResult} />
          ) : (
            <div className="empty-state">
              <p>
                Nhập dữ liệu và nhấn <strong>Dự đoán</strong>
              </p>
              <span>
                Kết quả của cả bốn mô hình sẽ hiện ở đây, mô hình được khuyên
                dùng sẽ được làm nổi bật.
              </span>
            </div>
          )}
        </section>
      </main>
    </div>
  );
}
