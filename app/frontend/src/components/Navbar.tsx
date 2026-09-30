// src/components/Navbar.tsx
import React from "react";
import { Activity, Database } from "lucide-react";

export const Navbar: React.FC = () => {
  return (
    <header className="bg-white border-b border-gray-200 shadow-sm">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="bg-[#009485] p-2 rounded-lg text-white">
            <Activity className="h-6 w-6" />
          </div>
          <div>
            <h1 className="text-lg font-bold text-gray-900">
              Diabetes AI Predictor
            </h1>
            <p className="text-xs text-gray-500">
              Hệ thống phân loại bệnh tiểu đường
            </p>
          </div>
        </div>
        <div className="flex items-center space-x-4">
          <span className="inline-flex items-center text-sm font-medium text-[#009485] bg-teal-50 px-3 py-1 rounded-full border border-teal-200">
            <Database className="w-4 h-4 mr-1.5" /> Pima Dataset
          </span>
        </div>
      </div>
    </header>
  );
};
