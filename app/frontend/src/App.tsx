// src/App.tsx
import { useState } from "react";
import { Navbar } from "./components/Navbar";
import { PredictionForm } from "./components/PredictionForm";
import { ResultComparison } from "./components/ResultComparison";
import { HistoryTable } from "./components/HistoryTable";
import { predictDiabetes } from "./services/api";
import type { DiabetesInput, PredictionResponse } from "./types/diabetes";

export default function App() {
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [latestResult, setLatestResult] = useState<PredictionResponse | null>(
    null,
  );
  const [history, setHistory] = useState<PredictionResponse[]>([]);

  const handlePredict = async (inputData: DiabetesInput) => {
    setIsLoading(true);
    try {
      const response = await predictDiabetes(inputData);
      setLatestResult(response);
      setHistory((prev) => [response, ...prev]);
    } catch (error) {
      console.error("Lỗi khi gọi API dự đoán:", error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app-shell min-h-screen text-slate-900">
      <Navbar />

      <main className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
        <section className="hero-panel mb-8 overflow-hidden rounded-[28px] border border-white/60 p-6 shadow-[0_20px_60px_rgba(15,118,110,0.12)] sm:p-8">
          <div className="flex flex-col gap-6 lg:flex-row lg:items-end lg:justify-between">
            <div className="max-w-2xl">
              <div className="mb-3 inline-flex items-center gap-2 rounded-full border border-emerald-200 bg-emerald-50 px-3 py-1 text-xs font-semibold uppercase tracking-[0.18em] text-emerald-700">
                <span className="h-2 w-2 rounded-full bg-emerald-500" />
                AI Health Intelligence
              </div>
              <h2 className="text-3xl font-black tracking-tight text-slate-900 sm:text-4xl">
                Hệ thống đánh giá nguy cơ tiểu đường
              </h2>
              <p className="mt-3 max-w-xl text-sm text-slate-600 sm:text-base">
                Ứng dụng tích hợp đồng thời 4 mô hình Machine Learning để đánh
                giá nguy cơ tiểu đường dựa trên chỉ số sức khỏe bệnh nhân.
              </p>
            </div>

            <div className="grid grid-cols-3 gap-3 sm:min-w-[320px]">
              <div className="stat-card">
                <div className="stat-value">4</div>
                <div className="stat-label">Mô hình</div>
              </div>
              <div className="stat-card">
                <div className="stat-value">AI</div>
                <div className="stat-label">Phân tích</div>
              </div>
              <div className="stat-card">
                <div className="stat-value">24s</div>
                <div className="stat-label">Xử lý</div>
              </div>
            </div>
          </div>
        </section>

        <div className="grid grid-cols-1 gap-8">
          <PredictionForm onSubmit={handlePredict} isLoading={isLoading} />

          {latestResult ? (
            <ResultComparison data={latestResult} />
          ) : (
            <div className="empty-panel rounded-[24px] border border-dashed border-emerald-200 bg-white/80 p-8 text-center text-slate-500 shadow-sm backdrop-blur-sm">
              <p className="text-base font-medium text-slate-600">
                Hãy nhập dữ liệu và nhấn{" "}
                <span className="font-bold text-emerald-600">
                  Chạy Dự Đoán 4 Model
                </span>
              </p>
              <p className="mt-2 text-sm text-slate-500">
                Kết quả phân tích so sánh từ 4 mô hình sẽ hiển thị ngay tại đây.
              </p>
            </div>
          )}

          <HistoryTable
            history={history}
            onSelectRecord={(record) => setLatestResult(record)}
            onClearHistory={() => {
              setHistory([]);
              setLatestResult(null);
            }}
          />
        </div>
      </main>
    </div>
  );
}
