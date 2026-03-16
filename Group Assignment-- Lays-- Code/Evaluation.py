# Member 5: Credit Card Fraud Detection Model Validation and Evaluation
# Core Functions: Load Member 4's PKL models → Predict test set → Calculate comprehensive evaluation metrics → Export results
import os
import joblib
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')  # Ignore irrelevant warnings

# Visualization & Evaluation Libraries
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)

# Configure matplotlib to avoid font garbling (ensure English text display normally)
plt.rcParams['font.family'] = 'DejaVu Sans'  # Universal English font
plt.rcParams['axes.unicode_minus'] = False  # Fix negative sign display issue

# ===================== Configure Paths =====================
# Input Files: Preprocessed test set from Member 3 (stored in 01_Input_Data folder)
X_TEST_PATH = "../01_Input_Data/x_test.csv"
Y_TEST_PATH = "../01_Input_Data/y_test.csv"

# Input Files: Trained models from Member 4 (stored in 01_Input_Data folder)
LR_MODEL_PATH = "../01_Input_Data/lr_model.pkl"  # Logistic Regression model
XGB_MODEL_PATH = "../01_Input_Data/xgb_model.pkl"  # XGBoost model

# Output Files: Evaluation results (stored in 03_Results folder)
METRICS_REPORT_PATH = "../03_Results/model_evaluation_metrics.csv"
PREDICTIONS_PATH = "../03_Results/test_set_predictions.csv"
CONFUSION_MATRIX_PATH = "../03_Results/confusion_matrix.png"


# ===================== 1. Load Data =====================
def load_data():
    """Load test set data and perform basic validation"""
    try:
        X_test = pd.read_csv(X_TEST_PATH)
        y_test = pd.read_csv(Y_TEST_PATH)
        # Adapt to Member 3's data format: convert y_test to 1D array if it's DataFrame
        if len(y_test.columns) == 1:
            y_test = y_test.iloc[:, 0].values
        print(f"✅ Test set loaded successfully")
        print(f"   - Test set feature dimension: {X_test.shape}")
        print(f"   - Test set label dimension: {y_test.shape}")
        print(f"   - Fraud sample ratio in test set: {round(np.sum(y_test) / len(y_test) * 100, 4)}%")
        return X_test, y_test
    except FileNotFoundError as e:
        print(f"❌ Data file not found: {e}")
        print("   Please ensure x_test.csv/y_test.csv are placed in the 01_Input_Data folder")
        return None, None


# ===================== 2. Load Models =====================
def load_models():
    """Load LR/XGBoost models trained by Member 4"""
    models = {}
    # Load Logistic Regression model
    try:
        lr_model = joblib.load(LR_MODEL_PATH)
        models["Logistic Regression"] = lr_model
        print(f"✅ Logistic Regression model loaded successfully ({LR_MODEL_PATH})")
    except FileNotFoundError:
        print(f"❌ Logistic Regression model file not found: {LR_MODEL_PATH}")
        print("   Please ensure Member 4's lr_model.pkl is placed in the 01_Input_Data folder")

    # Load XGBoost model
    try:
        xgb_model = joblib.load(XGB_MODEL_PATH)
        models["XGBoost"] = xgb_model
        print(f"✅ XGBoost model loaded successfully ({XGB_MODEL_PATH})")
    except FileNotFoundError:
        print(f"❌ XGBoost model file not found: {XGB_MODEL_PATH}")
        print("   Please ensure Member 4's xgb_model.pkl is placed in the 01_Input_Data folder")

    return models


# ===================== 3. Model Evaluation =====================
def evaluate_models(X_test, y_test, models):
    """Perform prediction for each model and calculate comprehensive evaluation metrics"""
    all_metrics = []
    all_predictions = pd.DataFrame({"True_Label": y_test})

    for model_name, model in models.items():
        print(f"\n===== Start Evaluating [{model_name}] ======")
        # Model prediction
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of fraud class (1)

        # Calculate core evaluation metrics (focus on F1/Recall for imbalanced data)
        metrics = {
            "Model_Name": model_name,
            "Accuracy": round(accuracy_score(y_test, y_pred), 4),
            "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "Recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
            "F1_Score": round(f1_score(y_test, y_pred, zero_division=0), 4),
            "AUC_ROC": round(roc_auc_score(y_test, y_pred_proba), 4)
        }
        all_metrics.append(metrics)

        # Save prediction results
        all_predictions[f"{model_name}_Predicted_Label"] = y_pred
        all_predictions[f"{model_name}_Fraud_Probability"] = y_pred_proba

        # Print detailed classification report
        print(f"\n{model_name} Detailed Classification Report:")
        print(classification_report(y_test, y_pred, target_names=["Normal Transaction", "Fraud Transaction"]))

        # Plot confusion matrix (only save XGBoost's)
        if model_name == "XGBoost":
            plot_confusion_matrix(y_test, y_pred, model_name)

    return all_metrics, all_predictions


# ===================== 4. Visualization: Confusion Matrix =====================
def plot_confusion_matrix(y_test, y_pred, model_name):
    """Plot and save confusion matrix (English labels only)"""
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    # Use pure English labels for axes
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal Transaction", "Fraud Transaction"],
                yticklabels=["Normal Transaction", "Fraud Transaction"])
    plt.title(f"{model_name} Confusion Matrix (Test Set)", fontsize=12, fontweight='bold')
    plt.xlabel("Predicted Label", fontsize=10, fontweight='medium')
    plt.ylabel("True Label", fontsize=10, fontweight='medium')
    plt.tight_layout()  # Ensure no text is cut off
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✅ Confusion matrix saved to: {CONFUSION_MATRIX_PATH}")


# ===================== 5. Export Results =====================
def export_results(all_metrics, all_predictions):
    """Export evaluation metrics and prediction results to 03_Results folder"""
    # Export evaluation metrics summary
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(METRICS_REPORT_PATH, index=False, encoding="utf-8-sig")
    print(f"\n✅ Model evaluation metrics exported to: {METRICS_REPORT_PATH}")

    # Export prediction results
    all_predictions.to_csv(PREDICTIONS_PATH, index=False, encoding="utf-8-sig")
    print(f"✅ Test set prediction results exported to: {PREDICTIONS_PATH}")


# ===================== Main Execution =====================
if __name__ == "__main__":
    print("===== Member 5: Credit Card Fraud Detection Model Validation Started =====")

    # 1. Load test set data
    X_test, y_test = load_data()
    if X_test is None or y_test is None:
        exit()  # Terminate if data loading fails

    # 2. Load Member 4's models
    models = load_models()
    if len(models) == 0:
        print("❌ No available models, task terminated")
        exit()

    # 3. Model evaluation
    all_metrics, all_predictions = evaluate_models(X_test, y_test, models)

    # 4. Export results
    export_results(all_metrics, all_predictions)

    print("\n===== Member 5 Task Completed! All results saved to 03_Results folder =====")