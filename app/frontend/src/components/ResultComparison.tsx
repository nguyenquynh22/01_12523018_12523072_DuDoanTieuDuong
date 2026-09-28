// src/components/ResultComparison.tsx
import React from 'react';
import type { PredictionResponse } from '../types/diabetes';
import { Award, CheckCircle2, AlertTriangle } from 'lucide-react';

interface ResultComparisonProps {
  data: PredictionResponse;
}

export const ResultComparison: React.FC<ResultComparisonProps> = ({ data }) => {
  return (
    <div className="bg-white rounded-xl shadow-md p-6 border border-gray-100 mt-6">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-xl font-bold text-gray-800 flex items-center">
            <span className="w-2 h-6 bg-[#009485] rounded-full mr-2"></span>
            Kết quả so sánh đồng thời 4 mô hình
          </h2>
          <p className="text-sm text-gray-500 mt-1">Mã phiên: {data.id} | Thời gian: {data.createdAt}</p>
        </div>
        <div className="bg-teal-50 border border-[#009485] text-[#009485] px-4 py-2 rounded-lg text-sm font-semibold flex items-center">
          <Award className="w-4 h-4 mr-1.5 text-[#009485]" />
          Khuyên dùng: {data.bestModel}
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {data.results.map((res, index) => {
          const isRisky = res.prediction === 1;
          const isBest = res.modelName === data.bestModel;

          return (
            <div 
              key={index} 
              className={`border rounded-xl p-5 flex flex-col justify-between transition-all ${
                isBest ? 'border-[#009485] bg-teal-50/30 ring-2 ring-[#009485]/20' : 'border-gray-200 bg-white'
              }`}
            >
              <div>
                <div className="flex items-center justify-between mb-3">
                  <span className="text-xs font-bold uppercase tracking-wider text-gray-500">Model {index + 1}</span>
                  {isBest && <span className="bg-[#009485] text-white text-[10px] px-2 py-0.5 rounded-full font-bold">Best</span>}
                </div>
                <h3 className="font-bold text-gray-900 text-base mb-2">{res.modelName}</h3>
                
                <div className={`inline-flex items-center px-2.5 py-1 rounded-full text-xs font-medium mb-3 ${
                  isRisky ? 'bg-red-100 text-red-800' : 'bg-green-100 text-green-800'
                }`}>
                  {isRisky ? <AlertTriangle className="w-3.5 h-3.5 mr-1" /> : <CheckCircle2 className="w-3.5 h-3.5 mr-1" />}
                  {res.statusText}
                </div>

                <p className="text-xs text-gray-600 mb-4">{res.description}</p>
              </div>

              <div className="border-t border-gray-100 pt-3 mt-2">
                <div className="flex justify-between items-center text-sm">
                  <span className="text-gray-500">Độ tin cậy:</span>
                  <span className="font-bold text-gray-900">{res.probability}%</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2 mt-1.5 overflow-hidden">
                  <div 
                    className={`h-2 rounded-full ${isRisky ? 'bg-amber-500' : 'bg-[#009485]'}`}
                    style={{ width: `${res.probability}%` }}
                  ></div>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};