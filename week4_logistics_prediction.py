"""
Week 4: Predictive Modeling and Optimization in Logistics Systems
Dataset: DataCoSupplyChainDataset.csv
Target: Days for shipping (real) — actual shipping duration in days.

Run from the folder containing DataCoSupplyChainDataset.csv:
    python week4_logistics_prediction.py

Outputs:
    week4_outputs/model_metrics.csv
    week4_outputs/prediction_scatter.png
    week4_outputs/feature_importance.png
    week4_outputs/actual_vs_predicted_sample.csv
    week4_outputs/optimization_recommendations.txt

Leakage prevention:
- Delivery Status and Late_delivery_risk are excluded because they can reveal
  information closely tied to the actual delivery outcome.
- The target is not used as an input feature.
"""

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.inspection import permutation_importance

DATA_FILE = Path("DataCoSupplyChainDataset.csv")
OUT_DIR = Path("week4_outputs")
OUT_DIR.mkdir(exist_ok=True)

if not DATA_FILE.exists():
    raise FileNotFoundError(
        "Dataset not found. Put DataCoSupplyChainDataset.csv in the same folder "
        "as this script, then run it again."
    )

# The DataCo dataset commonly uses Latin-1 encoding.
df = pd.read_csv(DATA_FILE, encoding="latin1", low_memory=False)
df.columns = df.columns.str.strip()

target = "Days for shipping (real)"
if target not in df.columns:
    raise KeyError(
        f"Target column '{target}' not found. Available columns include: "
        + ", ".join(df.columns[:20])
    )

# Keep only rows where the target is available and numeric.
df[target] = pd.to_numeric(df[target], errors="coerce")
df = df.dropna(subset=[target]).copy()

# Features available for a practical baseline prediction.
# Exclude direct outcome/status fields to reduce target leakage.
candidate_features = [
    "Days for shipment (scheduled)",
    "Shipping Mode",
    "Market",
    "Order Region",
    "Order Country",
    "Order Item Quantity",
    "Sales",
    "Product Price",
    "Category Name",
    "Department Name",
    "Customer Segment",
]
features = [c for c in candidate_features if c in df.columns]
if len(features) < 3:
    raise ValueError(
        "Not enough expected feature columns found. Inspect the CSV column names "
        "and update candidate_features in this script."
    )

X = df[features].copy()
y = df[target].copy()

# Convert numeric-looking fields safely; categorical fields remain strings.
numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
categorical_features = [c for c in X.columns if c not in numeric_features]

for c in numeric_features:
    X[c] = pd.to_numeric(X[c], errors="coerce")
for c in categorical_features:
    X[c] = X[c].astype("string").fillna("Missing").astype(str)

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
])
categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=10)),
])
preprocessor = ColumnTransformer([
    ("num", numeric_pipe, numeric_features),
    ("cat", categorical_pipe, categorical_features),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# Baseline decision tree and ensemble random forest.
models = {
    "Decision Tree": DecisionTreeRegressor(
        max_depth=12, min_samples_leaf=20, random_state=42
    ),
    "Random Forest": RandomForestRegressor(
        n_estimators=120, max_depth=18, min_samples_leaf=5,
        random_state=42, n_jobs=-1
    ),
}

results = []
fitted = {}
for name, regressor in models.items():
    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", regressor),
    ])
    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    mae = float(mean_absolute_error(y_test, pred))
    r2 = float(r2_score(y_test, pred))
    results.append({
        "Model": name,
        "MAE_days": mae,
        "RMSE_days": rmse,
        "R_squared": r2,
    })
    fitted[name] = pipeline
    print(f"\n{name}")
    print(f"MAE:  {mae:.3f} days")
    print(f"RMSE: {rmse:.3f} days")
    print(f"R²:   {r2:.3f}")

# Select model with lowest RMSE for the primary result.
metrics = pd.DataFrame(results).sort_values("RMSE_days")
metrics.to_csv(OUT_DIR / "model_metrics.csv", index=False)
best_name = str(metrics.iloc[0]["Model"])
best_model = fitted[best_name]
best_pred = best_model.predict(X_test)

print(f"\nSelected model by lowest test RMSE: {best_name}")

# Optional compact hyperparameter tuning on a training subset for runtime practicality.
# RandomizedSearchCV uses 3-fold CV; final test set remains untouched.
print("\nRunning a small 3-fold RandomizedSearchCV for Random Forest...")
rf_pipe = Pipeline([
    ("preprocess", preprocessor),
    ("model", RandomForestRegressor(random_state=42, n_jobs=-1)),
])
param_dist = {
    "model__n_estimators": [80, 120, 160],
    "model__max_depth": [10, 14, 18, None],
    "model__min_samples_leaf": [2, 5, 10],
    "model__max_features": [0.7, 1.0],
}
# Sample training data to keep tuning manageable on a large CSV.
sample_n = min(25000, len(X_train))
X_tune = X_train.sample(n=sample_n, random_state=42)
y_tune = y_train.loc[X_tune.index]
cv = KFold(n_splits=3, shuffle=True, random_state=42)
search = RandomizedSearchCV(
    rf_pipe, param_distributions=param_dist, n_iter=6,
    scoring="neg_root_mean_squared_error", cv=cv,
    random_state=42, n_jobs=-1, verbose=0
)
search.fit(X_tune, y_tune)
tuned_pred = search.best_estimator_.predict(X_test)
tuned_metrics = {
    "Model": "Tuned Random Forest",
    "MAE_days": float(mean_absolute_error(y_test, tuned_pred)),
    "RMSE_days": float(np.sqrt(mean_squared_error(y_test, tuned_pred))),
    "R_squared": float(r2_score(y_test, tuned_pred)),
}
metrics = pd.concat([metrics, pd.DataFrame([tuned_metrics])], ignore_index=True)
metrics.to_csv(OUT_DIR / "model_metrics.csv", index=False)
print("Best tuning parameters:", search.best_params_)
print("Tuned Random Forest metrics:", tuned_metrics)

# Actual vs predicted plot for the best tuned model.
final_model = search.best_estimator_
final_pred = tuned_pred
sample_size = min(3000, len(y_test))
rng = np.random.RandomState(42)
sample_idx = rng.choice(len(y_test), size=sample_size, replace=False)
plt.figure(figsize=(8, 6))
plt.scatter(y_test.iloc[sample_idx], final_pred[sample_idx], alpha=0.35, s=18)
low = min(float(y_test.min()), float(np.min(final_pred)))
high = max(float(y_test.max()), float(np.max(final_pred)))
plt.plot([low, high], [low, high], linestyle="--")
plt.xlabel("Actual shipping days")
plt.ylabel("Predicted shipping days")
plt.title("Actual vs Predicted Shipping Time — Tuned Random Forest")
plt.tight_layout()
plt.savefig(OUT_DIR / "prediction_scatter.png", dpi=200)
plt.close()

# Permutation importance on a manageable test sample; importance is at input-column level.
imp_n = min(5000, len(X_test))
X_imp = X_test.sample(n=imp_n, random_state=42)
y_imp = y_test.loc[X_imp.index]
perm = permutation_importance(
    final_model, X_imp, y_imp, n_repeats=3,
    random_state=42, scoring="neg_mean_absolute_error", n_jobs=-1
)
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": perm.importances_mean
}).sort_values("Importance", ascending=False)
importance.to_csv(OUT_DIR / "feature_importance.csv", index=False)

top = importance.head(10).sort_values("Importance")
plt.figure(figsize=(9, 6))
plt.barh(top["Feature"], top["Importance"])
plt.xlabel("Permutation importance (change in negative MAE score)")
plt.title("Top Features for Shipping-Time Prediction")
plt.tight_layout()
plt.savefig(OUT_DIR / "feature_importance.png", dpi=200)
plt.close()

pd.DataFrame({
    "Actual_shipping_days": y_test,
    "Predicted_shipping_days": final_pred,
    "Absolute_error_days": np.abs(y_test.to_numpy() - final_pred),
}).head(2000).to_csv(OUT_DIR / "actual_vs_predicted_sample.csv", index=False)

# Evidence-based recommendations from measured model outputs.
best_row = metrics.loc[metrics["Model"] == "Tuned Random Forest"].iloc[0]
top_features = importance.head(3)["Feature"].tolist()
recommendations = [
    "1. Use predicted shipping duration as a planning aid, not as a guaranteed delivery promise.",
    "2. Prioritize manual review for orders with high predicted duration or high prediction error.",
    "3. Investigate the top model features before changing operations: " + ", ".join(top_features) + ".",
    "4. Compare prediction errors by shipping mode, market and region to identify service-level gaps.",
    "5. Re-train and monitor the model periodically as order mix and operations change.",
    "6. Avoid claiming that a feature causes delay; feature importance indicates predictive usefulness, not causation.",
    f"7. Tuned Random Forest test MAE: {best_row['MAE_days']:.3f} days; "
    f"RMSE: {best_row['RMSE_days']:.3f} days; R-squared: {best_row['R_squared']:.3f}.",
]
(OUT_DIR / "optimization_recommendations.txt").write_text(
    "\n".join(recommendations), encoding="utf-8"
)

print(f"\nCompleted. Results saved in: {OUT_DIR.resolve()}")
print("Open model_metrics.csv and the PNG files inside week4_outputs.")
