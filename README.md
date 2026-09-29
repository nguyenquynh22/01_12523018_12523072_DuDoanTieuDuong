# Diabetes Prediction API

Diabetes risk prediction from eight health measurements. The application has a React/Vite frontend, FastAPI backend, FastAPI inference service, and optional MongoDB Atlas history storage.

## Public test URLs

Raw URLs for a spreadsheet or API client:

- Frontend: https://unaltered-eagle-wackiness.ngrok-free.dev/
- Backend health: https://unaltered-eagle-wackiness.ngrok-free.dev/health
- Prediction API (POST): https://unaltered-eagle-wackiness.ngrok-free.dev/api/v1/predict-disease

The prediction endpoint requires a POST request with a JSON body. Use PowerShell, Postman, or Insomnia; opening the endpoint in a browser does not send a POST request.

## Start locally

Start Docker Desktop, open PowerShell in the project root, and run:

```powershell
docker compose up --build -d
docker compose ps
```

Open the gateway at http://localhost:8080/, frontend directly at http://localhost:3000/, backend Swagger at http://localhost:8000/docs, and AI service Swagger at http://localhost:8001/docs.

## Publish through Ngrok

Leave Docker running. Open another PowerShell window in the project root:

```powershell
& "$PWD/ngrok.exe" http --domain=unaltered-eagle-wackiness.ngrok-free.dev 8080
```

Keep Ngrok running during the test. If the reserved domain is unavailable, run `& "$PWD/ngrok.exe" http 8080` and use the Forwarding URL Ngrok prints.

## Call the prediction API

Endpoint:

```text
POST /api/v1/predict-disease
Content-Type: application/json
```

Example request and local call:

```powershell
$body = @{
  pregnancies = 2
  glucose = 120
  bloodPressure = 70
  skinThickness = 20
  insulin = 80
  bmi = 25.5
  diabetesPedigreeFunction = 0.5
  age = 30
} | ConvertTo-Json

Invoke-RestMethod -Method Post `
  -Uri 'http://localhost:8000/api/v1/predict-disease' `
  -ContentType 'application/json' `
  -Body $body | ConvertTo-Json -Depth 6
```

Public call through Ngrok:

```powershell
$publicUrl = "https://unaltered-eagle-wackiness.ngrok-free.dev"
Invoke-RestMethod -Method Post `
  -Uri "$publicUrl/api/v1/predict-disease" `
  -ContentType 'application/json' `
  -Body $body | ConvertTo-Json -Depth 6
```

## Training and prediction logic

`ai-models/colab/03_train.ipynb` and `ai-models/src/train.py` use the same pipeline:

1. Split the stratified dataset into 80% train and 20% test (`random_state=42`).
2. Compare Logistic Regression, SVM, Naive Bayes, and Random Forest with 5-fold stratified cross-validation on the training split. For each model, select a probability threshold from 5-fold out-of-fold train predictions that reaches the illustrative 80% sensitivity target and maximizes specificity. Recommend the model with highest CV specificity among those meeting the target; report weighted F1 and positive-class recall as supporting metrics.
3. Fit each candidate model on the full 80% train split for inference. The held-out 20% test split is used only for final metrics.
4. At inference, run all four fitted models in parallel, each with its threshold selected on train. The API returns four predictions and highlights the model recommended by cross-validation.

After training, place the matching `model.joblib` (containing all four fitted models) and `decision_config.json` in `ai-models/models/`, then restart the AI service.

## Successful response

A valid request returns HTTP `200`. `results` contains predictions and CV/test metrics for all four models. `recommendedModel` identifies the model selected by CV specificity subject to the sensitivity target. Each model result includes its CV-selected threshold, positive_recall, and specificity. The values below illustrate the response structure and may differ for trained artifacts. The 80% target is only an educational screening assumption, not a clinically established target.

```json
{
  "id": "pred-example",
  "createdAt": "example",
  "inputData": {
    "pregnancies": 2,
    "glucose": 120,
    "bloodPressure": 70,
    "skinThickness": 20,
    "insulin": 80,
    "bmi": 25.5,
    "diabetesPedigreeFunction": 0.5,
    "age": 30
  },
  "recommendedModel": "Support Vector Machine (SVM)",
  "targetSensitivity": 0.8,
  "cvSampleCount": 614,
  "testSampleCount": 154,
  "results": [
    {
      "modelName": "Logistic Regression",
      "prediction": 0,
      "probability": 13.6,
      "threshold": 28.1,
      "statusText": "Bình thường",
      "description": "Dự đoán theo threshold được chọn từ CV trên train.",
      "cvMetrics": {
        "threshold": 0.280588297754102,
        "accuracy": 0.7296416938110749,
        "weighted_precision": 0.767949776338935,
        "weighted_recall": 0.7296416938110749,
        "weighted_f1_score": 0.7359380934920492,
        "positive_recall": 0.8037383177570093,
        "specificity": 0.69,
        "predicted_positive_count": 296
      },
      "testMetrics": {
        "accuracy": 0.7272727272727273,
        "weighted_precision": 0.7747515642252483,
        "weighted_recall": 0.7272727272727273,
        "weighted_f1_score": 0.7334710743801652,
        "positive_recall": 0.8333333333333334,
        "specificity": 0.67,
        "predicted_positive_count": 78
      }
    },
    {
      "modelName": "Support Vector Machine (SVM)",
      "prediction": 0,
      "probability": 18.6,
      "threshold": 25.9,
      "statusText": "Bình thường",
      "description": "Dự đoán theo threshold được chọn từ CV trên train.",
      "cvMetrics": {
        "threshold": 0.25872317618043317,
        "accuracy": 0.737785016286645,
        "weighted_precision": 0.7727615449636194,
        "weighted_recall": 0.737785016286645,
        "weighted_f1_score": 0.7438126250080237,
        "positive_recall": 0.8037383177570093,
        "specificity": 0.7025,
        "predicted_positive_count": 291
      },
      "testMetrics": {
        "accuracy": 0.6883116883116883,
        "weighted_precision": 0.7302916381863751,
        "weighted_recall": 0.6883116883116883,
        "weighted_f1_score": 0.6954236774461493,
        "positive_recall": 0.7592592592592593,
        "specificity": 0.65,
        "predicted_positive_count": 76
      }
    },
    {
      "modelName": "Naive Bayes",
      "prediction": 0,
      "probability": 2.3,
      "threshold": 17.7,
      "statusText": "Bình thường",
      "description": "Dự đoán theo threshold được chọn từ CV trên train.",
      "cvMetrics": {
        "threshold": 0.1770655664061519,
        "accuracy": 0.7166123778501629,
        "weighted_precision": 0.76039967260802,
        "weighted_recall": 0.7166123778501629,
        "weighted_f1_score": 0.7232698063546346,
        "positive_recall": 0.8037383177570093,
        "specificity": 0.67,
        "predicted_positive_count": 304
      },
      "testMetrics": {
        "accuracy": 0.6818181818181818,
        "weighted_precision": 0.7318757192174913,
        "weighted_recall": 0.6818181818181818,
        "weighted_f1_score": 0.6889952153110048,
        "positive_recall": 0.7777777777777778,
        "specificity": 0.63,
        "predicted_positive_count": 79
      }
    },
    {
      "modelName": "Random Forest",
      "prediction": 0,
      "probability": 10.0,
      "threshold": 32.0,
      "statusText": "Bình thường",
      "description": "Dự đoán theo threshold được chọn từ CV trên train.",
      "cvMetrics": {
        "threshold": 0.32,
        "accuracy": 0.7345276872964169,
        "weighted_precision": 0.7708275647894278,
        "weighted_recall": 0.7345276872964169,
        "weighted_f1_score": 0.7406665352674943,
        "positive_recall": 0.8037383177570093,
        "specificity": 0.6975,
        "predicted_positive_count": 293
      },
      "testMetrics": {
        "accuracy": 0.7142857142857143,
        "weighted_precision": 0.7561692693271641,
        "weighted_recall": 0.7142857142857143,
        "weighted_f1_score": 0.7208050376589703,
        "positive_recall": 0.7962962962962963,
        "specificity": 0.67,
        "predicted_positive_count": 76
      }
    }
  ]
}
```

The model thresholds are not fixed at 0.5: they are selected using out-of-fold predictions on the 80% train split to meet the sensitivity target while maximizing specificity. Test metrics are calculated afterward and do not select models or thresholds. The target must be chosen with clinical input for any real screening use; this project dataset and validation are not sufficient for clinical deployment.

## Health, invalid input, and logs

```powershell
Invoke-RestMethod 'http://localhost:8000/health' | ConvertTo-Json
$invalidBody = @{ glucose = 120 } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri 'http://localhost:8000/api/v1/predict-disease' `
  -ContentType 'application/json' -Body $invalidBody
docker compose logs -f --tail=100 backend ai-service nginx
```

- `200`: prediction succeeded.
- `422`: required JSON fields are missing or have invalid types.
- `502`: the AI service rejected or failed the inference request; inspect `ai-service` and `backend` logs.
- `503`: the backend cannot connect to the AI service, or model/config artifacts are missing.
- `500`: internal processing error; inspect backend logs.
- If Ngrok displays a warning or rate limit page, inspect the tunnel and http://localhost:4040/.

Stop Ngrok with `Ctrl+C`. Stop the containers when done:

```powershell
docker compose down
```

## Project structure

```text
app/frontend/          React + Vite
app/backend/           FastAPI gateway, AI service client, and MongoDB history
ai-models/service/     FastAPI inference service
ai-models/models/      model.joblib and decision_config.json
ai-models/src/train.py Local training script, aligned with Colab
ai-models/colab/03_train.ipynb Colab training notebook
nginx/default.conf     Gateway routing for frontend/backend/AI
 docker-compose.yml    Local services
```
