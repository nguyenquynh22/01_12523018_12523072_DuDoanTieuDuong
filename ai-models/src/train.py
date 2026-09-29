import json
import zipfile
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / 'data'
MODEL_DIR = ROOT_DIR / 'models'
TARGET_SENSITIVITY = 0.80  # Illustrative screening target; confirm with clinical stakeholders.
MODEL_LABELS = {
    'logistic': 'Logistic Regression',
    'svm': 'Support Vector Machine (SVM)',
    'naive_bayes': 'Naive Bayes',
    'random_forest': 'Random Forest',
}
ZERO_AS_MISSING = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']


def resolve_data_file():
    csv_path = DATA_DIR / 'diabetes.csv'
    if csv_path.exists():
        return csv_path
    archive_path = DATA_DIR / 'archive.zip'
    if archive_path.exists():
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
        csv_path = next(DATA_DIR.glob('*.csv'), None)
        if csv_path is not None:
            return csv_path
    raise FileNotFoundError(f'Expected diabetes.csv or archive.zip in {DATA_DIR}')


def build_pipeline(estimator):
    return Pipeline([
        ('impute_invalid_zero', ColumnTransformer(
            [('median', SimpleImputer(strategy='median', missing_values=0), ZERO_AS_MISSING)],
            remainder='passthrough',
        )),
        ('scaler', StandardScaler()),
        ('classifier', estimator),
    ])


def calculate_metrics(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true, predictions, average='weighted', zero_division=0
    )
    return {
        'accuracy': float(accuracy_score(y_true, predictions)),
        'weighted_precision': float(weighted_precision),
        'weighted_recall': float(weighted_recall),
        'weighted_f1_score': float(weighted_f1),
        'positive_recall': float(recall_score(y_true, predictions, pos_label=1, zero_division=0)),
        'specificity': float(recall_score(y_true, predictions, pos_label=0, zero_division=0)),
        'predicted_positive_count': int(predictions.sum()),
    }


def choose_screening_threshold(y_true, probabilities):
    thresholds = np.unique(np.concatenate(([0.0, 1.0], probabilities)))
    feasible = []
    for threshold in thresholds:
        metrics = calculate_metrics(y_true, probabilities, float(threshold))
        if metrics['positive_recall'] >= TARGET_SENSITIVITY:
            # At a fixed minimum sensitivity, maximize specificity and then weighted F1.
            feasible.append((metrics['specificity'], metrics['weighted_f1_score'], float(threshold), metrics))
    if not feasible:
        raise ValueError(f'No candidate threshold reaches sensitivity target {TARGET_SENSITIVITY:.0%}')
    specificity, weighted_f1, threshold, metrics = max(feasible)
    return threshold, metrics


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(resolve_data_file())
    feature_columns = [col for col in df.columns if col not in ('Outcome', 'index')]
    X, y = df[feature_columns], df['Outcome'].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    candidates = {
        'logistic': build_pipeline(LogisticRegression(random_state=42, max_iter=1000)),
        'svm': build_pipeline(SVC(probability=True, random_state=42)),
        'naive_bayes': build_pipeline(GaussianNB()),
        'random_forest': build_pipeline(RandomForestClassifier(n_estimators=100, random_state=42)),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = {}
    print(f'Evaluating four models with 5-fold CV and target sensitivity {TARGET_SENSITIVITY:.0%}...')
    for key, model in candidates.items():
        oof_probabilities = cross_val_predict(model, X_train, y_train, cv=cv, method='predict_proba')[:, 1]
        threshold, metrics = choose_screening_threshold(y_train, oof_probabilities)
        cv_results[key] = {'threshold': threshold, **metrics}

    # Recommend the model with highest cross-validated specificity while meeting the sensitivity target.
    selected_key = max(
        candidates,
        key=lambda key: (
            cv_results[key]['specificity'],
            cv_results[key]['weighted_f1_score'],
            cv_results[key]['positive_recall'],
        ),
    )

    model_metrics = {}
    for key, model in candidates.items():
        model.fit(X_train, y_train)
        threshold = cv_results[key]['threshold']
        test_probabilities = model.predict_proba(X_test)[:, 1]
        model_metrics[key] = {
            'threshold': threshold,
            'cv_metrics': cv_results[key],
            'test_metrics': calculate_metrics(y_test, test_probabilities, threshold),
        }

    decision_config = {
        'model_key': selected_key,
        'model_name': MODEL_LABELS[selected_key],
        'target_sensitivity': TARGET_SENSITIVITY,
        'selection_metric': 'highest_specificity_subject_to_target_sensitivity',
        'selection_method': 'Per-model threshold selected from 5-fold out-of-fold predictions on 80% train to attain at least the configured positive-class recall; recommend highest CV specificity.',
        'model_keys': list(candidates),
        'models': model_metrics,
        'cv_sample_count': int(len(y_train)),
        'test_sample_count': int(len(y_test)),
    }
    joblib.dump(candidates, MODEL_DIR / 'model.joblib', compress=3)
    (MODEL_DIR / 'decision_config.json').write_text(json.dumps(decision_config, indent=2), encoding='utf-8')

    metadata_path = MODEL_DIR / 'metadata.json'
    metadata = json.loads(metadata_path.read_text(encoding='utf-8')) if metadata_path.exists() else {}
    metadata.update({
        'metrics': {MODEL_LABELS[key]: values['test_metrics'] for key, values in model_metrics.items()},
        'training_date': date.today().isoformat(),
        'target_sensitivity': TARGET_SENSITIVITY,
        'train_sample_count': int(len(y_train)),
        'test_sample_count': int(len(y_test)),
        'selected_model': MODEL_LABELS[selected_key],
        'decision_config': decision_config,
        'validation_method': 'Stratified 80/20 split; thresholds selected using 5-fold out-of-fold training predictions; test split used only for final reporting.',
        'notes': 'The sensitivity target is an illustrative screening assumption, not clinically validated. External validation and clinical review are required before real-world use.',
    })
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    chosen = cv_results[selected_key]
    print(f"Recommended: {MODEL_LABELS[selected_key]} | threshold={chosen['threshold']:.4f} | CV sensitivity={chosen['positive_recall']:.3f} | CV specificity={chosen['specificity']:.3f}")


if __name__ == '__main__':
    main()
