// src/components/PredictionForm.tsx
import React, { useState } from "react";
import type { DiabetesInput } from "../types/diabetes";
import { Send, Zap } from "lucide-react"; // Đã lược bỏ Loader2 để tránh lỗi

interface PredictionFormProps {
  onSubmit: (data: DiabetesInput) => Promise<void>;
  isLoading: boolean;
}

const SAMPLE_CASES: DiabetesInput[] = [
  {
    pregnancies: 2,
    glucose: 120,
    bloodPressure: 70,
    skinThickness: 20,
    insulin: 79,
    bmi: 25.5,
    diabetesPedigreeFunction: 0.5,
    age: 33,
  },
  {
    pregnancies: 6,
    glucose: 148,
    bloodPressure: 72,
    skinThickness: 35,
    insulin: 0,
    bmi: 33.6,
    diabetesPedigreeFunction: 0.627,
    age: 50,
  },
  {
    pregnancies: 1,
    glucose: 85,
    bloodPressure: 66,
    skinThickness: 29,
    insulin: 0,
    bmi: 26.6,
    diabetesPedigreeFunction: 0.351,
    age: 31,
  },
  {
    pregnancies: 8,
    glucose: 183,
    bloodPressure: 64,
    skinThickness: 32,
    insulin: 0,
    bmi: 23.3,
    diabetesPedigreeFunction: 0.672,
    age: 32,
  },
];

export const PredictionForm: React.FC<PredictionFormProps> = ({
  onSubmit,
  isLoading,
}) => {
  const [formData, setFormData] = useState<DiabetesInput>(SAMPLE_CASES[0]);

  const handleFillSample = () => {
    const randomSample =
      SAMPLE_CASES[Math.floor(Math.random() * SAMPLE_CASES.length)];
    setFormData(randomSample);
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: Number(value) || 0,
    }));
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    void onSubmit(formData);
  };

  return (
    <div className="glass-panel rounded-[26px] border border-white/70 bg-white/80 p-5 shadow-[0_18px_45px_rgba(15,23,42,0.08)] backdrop-blur-sm sm:p-6">
      <div className="mb-5 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <h2 className="flex items-center text-xl font-bold text-slate-800">
          <span className="mr-2 h-6 w-2 rounded-full bg-gradient-to-b from-emerald-500 to-teal-600" />
          Nhập chỉ số y tế bệnh nhân
        </h2>

        <button
          type="button"
          onClick={handleFillSample}
          className="inline-flex items-center justify-center rounded-xl border border-emerald-200 bg-emerald-50 px-3 py-2 text-xs font-semibold text-emerald-700 transition hover:bg-emerald-100"
        >
          <Zap className="mr-1.5 h-3.5 w-3.5" />
          Mẫu ngẫu nhiên
        </button>
      </div>

      <form
        onSubmit={handleSubmit}
        className="grid grid-cols-1 gap-4 md:grid-cols-2"
      >
        {[
          { label: "Số lần mang thai", name: "pregnancies", type: "number" },
          { label: "Đường huyết (Glucose)", name: "glucose", type: "number" },
          {
            label: "Huyết áp (BloodPressure)",
            name: "bloodPressure",
            type: "number",
          },
          {
            label: "Độ dày da (SkinThickness)",
            name: "skinThickness",
            type: "number",
          },
          { label: "Insulin", name: "insulin", type: "number" },
          { label: "Chỉ số BMI", name: "bmi", type: "number", step: "0.1" },
          {
            label: "Hệ số di truyền",
            name: "diabetesPedigreeFunction",
            type: "number",
            step: "0.001",
          },
          { label: "Tuổi (Age)", name: "age", type: "number" },
        ].map((field) => (
          <div key={field.name} className="space-y-2">
            <label className="block text-sm font-medium text-slate-700">
              {field.label}
            </label>
            <input
              type={field.type}
              name={field.name}
              step={field.step}
              value={formData[field.name as keyof DiabetesInput]}
              onChange={handleChange}
              className="field-input mt-0 block w-full rounded-xl border border-slate-200 bg-white px-3 py-2.5 text-sm text-slate-800 shadow-sm outline-none transition focus:border-emerald-500 focus:ring-2 focus:ring-emerald-100"
              required
            />
          </div>
        ))}

        <div className="md:col-span-2 mt-2">
          <button
            type="submit"
            disabled={isLoading}
            className="flex w-full items-center justify-center rounded-xl bg-gradient-to-r from-emerald-600 to-teal-600 px-4 py-3.5 text-sm font-semibold text-white shadow-[0_16px_30px_rgba(13,148,136,0.28)] transition hover:from-emerald-700 hover:to-teal-700 disabled:cursor-not-allowed disabled:opacity-60"
          >
            {isLoading ? (
              <div className="flex items-center gap-2">
                <div className="h-5 w-5 animate-spin rounded-full border-2 border-white/70 border-t-transparent" />
                <span>Đang chạy đồng thời 4 mô hình AI...</span>
              </div>
            ) : (
              <>
                <Send className="mr-2 h-4 w-4" />
                Chạy Dự Đoán 4 Model
              </>
            )}
          </button>
        </div>
      </form>
    </div>
  );
};
