import React from "react";
import type { PredictionResponse } from "../types/diabetes";
import { CheckCircle2, AlertTriangle, Award } from "lucide-react";

interface ResultComparisonProps { data: PredictionResponse }

export const ResultComparison: React.FC<ResultComparisonProps> = ({ data }) => (
  <section className="mt-6 rounded-xl border border-gray-100 bg-white p-6 shadow-md">
    <div className="mb-5">
      <h2 className="flex items-center text-xl font-bold text-gray-800">
        <span className="mr-2 h-6 w-2 rounded-full bg-[#009485]" />So sánh kết quả 4 mô hình
      </h2>
      <p className="mt-1 text-sm text-gray-500">{data.createdAt} | Mục tiêu Recall lớp dương: {(data.targetSensitivity * 100).toFixed(0)}% | Model khuyên dùng: {data.recommendedModel}</p>
      <p className="mt-1 text-xs text-slate-500">Ngưỡng từng model được chọn trên CV để đạt Recall mục tiêu, sau đó ưu tiên Specificity cao nhất. Mục tiêu này là giả định mô phỏng, chưa được xác nhận lâm sàng.</p>
    </div>
    <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
      {data.results.map((result) => {
        const isRecommended = result.modelName === data.recommendedModel;
        const isRisky = result.prediction === 1;
        return (
          <article key={result.modelName} className={`rounded-xl border p-5 ${isRecommended ? "border-emerald-500 bg-emerald-50/50 ring-2 ring-emerald-500/20" : "border-slate-200 bg-white"}`}>
            <div className="mb-3 flex items-center justify-between gap-2">
              <h3 className="font-bold text-slate-900">{result.modelName}</h3>
              {isRecommended && <span className="inline-flex items-center rounded-full bg-emerald-600 px-2.5 py-1 text-xs font-semibold text-white"><Award className="mr-1 h-3.5 w-3.5" />Khuyên dùng</span>}
            </div>
            <div className={`mb-3 inline-flex items-center rounded-full px-2.5 py-1 text-xs font-medium ${isRisky ? "bg-red-100 text-red-800" : "bg-green-100 text-green-800"}`}>
              {isRisky ? <AlertTriangle className="mr-1 h-3.5 w-3.5" /> : <CheckCircle2 className="mr-1 h-3.5 w-3.5" />}{result.statusText}
            </div>
            <p className="text-sm text-slate-700">Xác suất lớp nguy cơ: <strong>{result.probability}%</strong> | Ngưỡng: {result.threshold}%</p>
            <div className="mt-2 h-2 overflow-hidden rounded-full bg-slate-100"><div className={`h-2 rounded-full ${isRisky ? "bg-red-500" : "bg-emerald-600"}`} style={{ width: `${result.probability}%` }} /></div>
            <div className="mt-4 border-t border-slate-200 pt-3 text-xs text-slate-600">
              <p className="font-semibold text-slate-800">5-fold CV trên train ({data.cvSampleCount} mẫu)</p>
              <p className="mt-1">Threshold {result.threshold}% | Accuracy {(result.cvMetrics.accuracy * 100).toFixed(1)}% | Weighted F1 {(result.cvMetrics.weighted_f1_score * 100).toFixed(1)}%</p>
              <p className="mt-1 font-medium text-amber-800">Positive recall: {(result.cvMetrics.positive_recall * 100).toFixed(1)}% | Specificity: {(result.cvMetrics.specificity * 100).toFixed(1)}%</p>
              <p className="mt-2 font-semibold text-slate-800">Test độc lập ({data.testSampleCount} mẫu)</p>
              <p className="mt-1">Accuracy {(result.testMetrics.accuracy * 100).toFixed(1)}% | Weighted F1 {(result.testMetrics.weighted_f1_score * 100).toFixed(1)}%</p>
              <p className="mt-1 font-medium text-amber-800">Positive recall: {(result.testMetrics.positive_recall * 100).toFixed(1)}% | Specificity: {(result.testMetrics.specificity * 100).toFixed(1)}%</p>
            </div>
          </article>
        );
      })}
    </div>
  </section>
);
