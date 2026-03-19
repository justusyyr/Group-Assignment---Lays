# ---------------- 1. Import all required libraries ----------------
import pandas as pd
import numpy as np
import time
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
import joblib

# Ignore irrelevant warnings for cleaner output
import warnings
warnings.filterwarnings("ignore")

# ---------------- 2. Load training data from your desktop (Path is 100% matched) ----------------
print("Loading training data...")
# Using raw string (r prefix) for Windows path compatibility
x_train_path = r"C:\Users\86180\Desktop\x_train_smote.csv"
y_train_path = r"C:\Users\86180\Desktop\y_train_smote.csv"

# Load features and labels
X_train = pd.read_csv(x_train_path)
y_train = pd.read_csv(y_train_path).values.ravel()  # Convert label to 1D array for model input

# Print dataset info to confirm successful loading
print(f"✅ Data loading complete!")
print(f"Training set features (rows/columns): {X_train.shape}")
print(f"Training set class distribution:\n{pd.Series(y_train).value_counts()}")
# ---------------- 3. Train Logistic Regression Model ----------------
print("\n🚀 Starting Logistic Regression training...")
start_time = time.time()

# Define model, fix random state for reproducibility
lr_model = LogisticRegression(random_state=42, max_iter=1000)

# Hyperparameter search space (Lightweight tuning, meets project requirements)
lr_param_grid = {
    "C": [0.01, 0.1, 1, 10, 100]  # Regularization strength, core tuning parameter
}

# 5-Fold Cross Validation to find best parameters, using F1-score (suitable for imbalanced data)
lr_grid_search = GridSearchCV(
    estimator=lr_model,
    param_grid=lr_param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1,
    verbose=1
)

# Execute training
lr_grid_search.fit(X_train, y_train)

# Extract best model and results
best_lr_model = lr_grid_search.best_estimator_
print(f"\n✅ Logistic Regression training complete! Time elapsed: {time.time() - start_time:.2f}s")
print(f"Logistic Regression best parameters: {lr_grid_search.best_params_}")
print(f"5-Fold CV best F1-Score: {lr_grid_search.best_score_:.4f}")
# ---------------- 4. Train XGBoost Model ----------------
print("\n🚀 Starting XGBoost training...")
start_time = time.time()

# Define model, fix random state, disable incompatible parameters
xgb_model = XGBClassifier(
    random_state=42,
    use_label_encoder=False,
    eval_metric="logloss"
)

# Hyperparameter search space (Strictly following project requirements, only core lightweight params)
xgb_param_grid = {
    "learning_rate": [0.05, 0.1],  # Step size shrinkage
    "max_depth": [3, 4, 5]         # Maximum tree depth, controls complexity
}

# 5-Fold Cross Validation to find best parameters
xgb_grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=xgb_param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1,
    verbose=1
)

# Execute training
xgb_grid_search.fit(X_train, y_train)

# Extract best model and results
best_xgb_model = xgb_grid_search.best_estimator_
print(f"\n✅ XGBoost training complete! Time elapsed: {time.time() - start_time:.2f}s")
print(f"XGBoost best parameters: {xgb_grid_search.best_params_}")
print(f"5-Fold CV best F1-Score: {xgb_grid_search.best_score_:.4f}")
# ---------------- 5. Save model files to desktop ----------------
print("\n💾 Saving model files to desktop...")

# Model save paths (Same directory as your data files, on the desktop)
lr_save_path = r"C:\Users\86180\Desktop\lr_model.pkl"
xgb_save_path = r"C:\Users\86180\Desktop\xgb_model.pkl"

# Save the two best models
joblib.dump(best_lr_model, lr_save_path)
joblib.dump(best_xgb_model, xgb_save_path)

print("✅ Models saved successfully!")
print(f"You can find these 2 files on your desktop:")
print(f"1. lr_model.pkl (Logistic Regression Model)")
print(f"2. xgb_model.pkl (XGBoost Model)")
