// src/components/HistoryTable.tsx
import React from "react";
import type { PredictionResponse } from "../types/diabetes";
import {
  History,
  Eye,
  Trash2,
  CheckCircle2,
  AlertTriangle,
} from "lucide-react";

interface HistoryTableProps {
  history: PredictionResponse[];
  onSelectRecord: (record: PredictionResponse) => void;
  onClearHistory: () => void;
}

export const HistoryTable: React.FC<HistoryTableProps> = ({
  history,
  onSelectRecord,
  onClearHistory,
}) => {
  if (history.length === 0) {
    return (
      <div className="rounded-[24px] border border-dashed border-slate-200 bg-white/80 p-6 text-center shadow-sm">
        <div className="mx-auto mb-3 flex h-12 w-12 items-center justify-center rounded-full bg-emerald-50 text-emerald-600">
          <History className="h-5 w-5" />
        </div>
        <h3 className="text-lg font-semibold text-slate-800">
          Chưa có lịch sử dự đoán
        </h3>
        <p className="mt-2 text-sm text-slate-500">
          Sau khi chạy dự đoán lần đầu, bạn có thể xem lại kết quả và chọn bản
          ghi ở đây.
        </p>
      </div>
    );
  }

  return (
    <div className="mt-8 rounded-[24px] border border-slate-200 bg-white/80 p-4 shadow-[0_18px_45px_rgba(15,23,42,0.08)] backdrop-blur-sm sm:p-6">
      <div className="mb-4 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <h2 className="flex items-center text-xl font-bold text-slate-800">
          <History className="mr-2 h-5 w-5 text-emerald-600" />
          Lịch sử các lần dự đoán gần đây ({history.length})
        </h2>

        <button
          onClick={onClearHistory}
          className="inline-flex items-center justify-center rounded-xl border border-red-200 bg-red-50 px-3 py-2 text-xs font-medium text-red-600 transition hover:bg-red-100"
        >
          <Trash2 className="mr-1.5 h-3.5 w-3.5" />
          Xóa lịch sử
        </button>
      </div>

      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-slate-200 text-sm">
          <thead className="bg-slate-50">
            <tr>
              <th className="px-4 py-3 text-left text-[11px] font-bold uppercase tracking-[0.12em] text-slate-500">
                Thời gian / ID
              </th>
              <th className="px-4 py-3 text-left text-[11px] font-bold uppercase tracking-[0.12em] text-slate-500">
                Chỉ số chính
              </th>
              <th className="px-4 py-3 text-left text-[11px] font-bold uppercase tracking-[0.12em] text-slate-500">
                Model tốt nhất
              </th>
              <th className="px-4 py-3 text-left text-[11px] font-bold uppercase tracking-[0.12em] text-slate-500">
                Kết quả
              </th>
              <th className="px-4 py-3 text-right text-[11px] font-bold uppercase tracking-[0.12em] text-slate-500">
                Thao tác
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-200 bg-white text-sm">
            {history.map((item) => {
              const bestResult =
                item.results.find((r) =>
                  r.modelName.includes("Random Forest"),
                ) || item.results[0];
              const isRisky = bestResult.prediction === 1;

              return (
                <tr key={item.id} className="transition hover:bg-slate-50">
                  <td className="px-4 py-4 align-top">
                    <div className="font-semibold text-slate-800">
                      {item.createdAt}
                    </div>
                    <div className="mt-1 text-xs text-slate-400">
                      ID: {item.id}
                    </div>
                  </td>
                  <td className="px-4 py-4 align-top text-slate-600">
                    <div className="flex flex-wrap gap-1.5">
                      <span className="rounded-md bg-slate-100 px-2 py-1 text-[11px]">
                        Glu: {item.inputData.glucose}
                      </span>
                      <span className="rounded-md bg-slate-100 px-2 py-1 text-[11px]">
                        BMI: {item.inputData.bmi}
                      </span>
                      <span className="rounded-md bg-slate-100 px-2 py-1 text-[11px]">
                        Age: {item.inputData.age}
                      </span>
                    </div>
                  </td>
                  <td className="px-4 py-4 align-top">
                    <span className="inline-flex rounded-full border border-emerald-200 bg-emerald-50 px-2.5 py-1 text-[11px] font-semibold text-emerald-700">
                      {item.bestModel}
                    </span>
                  </td>
                  <td className="px-4 py-4 align-top">
                    <span
                      className={`inline-flex items-center rounded-full px-2.5 py-1 text-[11px] font-medium ${
                        isRisky
                          ? "bg-red-100 text-red-700"
                          : "bg-green-100 text-green-700"
                      }`}
                    >
                      {isRisky ? (
                        <AlertTriangle className="mr-1 h-3.5 w-3.5" />
                      ) : (
                        <CheckCircle2 className="mr-1 h-3.5 w-3.5" />
                      )}
                      {bestResult.statusText} ({bestResult.probability}%)
                    </span>
                  </td>
                  <td className="px-4 py-4 text-right align-top">
                    <button
                      onClick={() => onSelectRecord(item)}
                      className="inline-flex items-center rounded-lg bg-emerald-50 px-3 py-1.5 text-xs font-medium text-emerald-700 transition hover:bg-emerald-100"
                    >
                      <Eye className="mr-1.5 h-3.5 w-3.5" />
                      Xem chi tiết
                    </button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
};
