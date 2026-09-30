export interface DiabetesInput {
  pregnancies: number;
  glucose: number;
  bloodPressure: number;
  skinThickness: number;
  insulin: number;
  bmi: number;
  diabetesPedigreeFunction: number;
  age: number;
}

export interface ModelPredictionResult {
  threshold: number;
  modelName: string;
  prediction: number;
  probability: number;
  statusText: string;
  description: string;
  cvMetrics: ModelMetrics;
  testMetrics: ModelMetrics;
}

export interface ModelMetrics {
  accuracy: number;
  weighted_precision: number;
  weighted_recall: number;
  weighted_f1_score: number;
  positive_recall: number;
  specificity: number;
  predicted_positive_count: number;
}

export interface PredictionResponse {
  id: string;
  createdAt: string;
  inputData: DiabetesInput;
  results: ModelPredictionResult[];
  recommendedModel: string;
  targetSensitivity: number;
  cvSampleCount: number;
  testSampleCount: number;
}
