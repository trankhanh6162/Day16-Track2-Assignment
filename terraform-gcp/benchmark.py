import json
import time
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


DATA_PATH = Path.home() / "ml-benchmark" / "creditcard.csv"
RESULT_PATH = Path.home() / "ml-benchmark" / "benchmark_result.json"
RANDOM_STATE = 42


load_started = time.perf_counter()
data = pd.read_csv(DATA_PATH)
load_seconds = time.perf_counter() - load_started

X = data.drop(columns="Class")
y = data["Class"]
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y,
)
model = lgb.LGBMClassifier(
    objective="binary",
    n_estimators=200,
    learning_rate=0.05,
    num_leaves=31,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbosity=-1,
)

training_started = time.perf_counter()
model.fit(
    X_train,
    y_train,
)
training_seconds = time.perf_counter() - training_started

test_probability = model.predict_proba(X_test)[:, 1]
test_prediction = (test_probability >= 0.5).astype(int)

single_row = X_test.iloc[[0]]
model.predict_proba(single_row)  # Warm up before measuring latency.
latency_repetitions = 100
latency_started = time.perf_counter()
for _ in range(latency_repetitions):
    model.predict_proba(single_row)
latency_ms = (
    (time.perf_counter() - latency_started) / latency_repetitions * 1000
)

batch = X_test.iloc[:1000]
throughput_started = time.perf_counter()
model.predict_proba(batch)
throughput_seconds = time.perf_counter() - throughput_started

results = {
    "dataset_rows": len(data),
    "dataset_features": X.shape[1],
    "fraud_rows": int(y.sum()),
    "load_data_seconds": load_seconds,
    "training_seconds": training_seconds,
    "best_iteration": int(model.n_estimators_),
    "auc_roc": roc_auc_score(y_test, test_probability),
    "accuracy": accuracy_score(y_test, test_prediction),
    "f1_score": f1_score(y_test, test_prediction),
    "precision": precision_score(y_test, test_prediction, zero_division=0),
    "recall": recall_score(y_test, test_prediction, zero_division=0),
    "inference_latency_ms_1_row": latency_ms,
    "inference_throughput_rows_per_second_1000_rows": (
        len(batch) / throughput_seconds
    ),
}

RESULT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
print(json.dumps(results, indent=2))
print(f"\nSaved results to {RESULT_PATH}")
