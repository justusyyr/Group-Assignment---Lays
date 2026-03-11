# ==============================================
# Credit Card Fraud Detection - Member 1: Data Loading and Initial Exploration
# Save path: CreditCardFraudDetection_GroupXX/code/01_data_loading_and_exploration.py
# Environment: Python3.9 + pandas1.5.3 + numpy1.24.3 + matplotlib3.7.1
# ==============================================
# Module 1: Import libraries and global settings (fix Chinese/negative sign display)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Matplotlib settings for English display (no Chinese needed, keep for plot quality)
plt.rcParams['axes.unicode_minus'] = False    # Fix negative sign display
plt.rcParams['figure.figsize'] = (6, 6)       # Default plot size
plt.rcParams['figure.dpi'] = 300                    # High resolution (300dpi)

# ==============================================
# Module 2: Load dataset and basic exploration
# ==============================================
# ********** Replace with your local creditcard.csv path **********
# Example (Windows): 'C:/CreditCardFraudDetection_GroupXX/creditcard.csv'
# Example (Mac): '/Users/xxx/CreditCardFraudDetection_GroupXX/creditcard.csv'
df = pd.read_csv('D:/524/creditcard.csv')

# View first 5 rows
print("===== First 5 Rows of Dataset =====")
print(df.head())
# Basic dataset info (dimension, data type, non-null count)
print("\n===== Basic Dataset Information =====")
df.info()
# Statistical description (mean, std, min/max)
print("\n===== Statistical Description =====")
print(df.describe())
# Feature column names
print("\n===== Feature Column Names =====")
print(df.columns.tolist())

# ==============================================
# Module 3: Check missing values, duplicates, outliers
# ==============================================
# 1. Missing values check
print("===== Missing Value Rate =====")
missing_rate = df.isnull().sum() / len(df)
print(missing_rate[missing_rate > 0])  # Empty if no missing values

# 2. Duplicates check and removal
print(f"\n===== Duplicate Values Processing =====")
print(f"Original dataset rows: {len(df)}")
df = df.drop_duplicates()  # Remove duplicate rows
print(f"Rows after removing duplicates: {len(df)}")
print(f"Number of duplicate rows: {284807 - len(df)}")

# 3. Outliers initial check
print(f"\n===== Label/Amount Outliers Check =====")
print(f"Class column unique values: {df['Class'].unique()}")
print(f"Amount column max: {df['Amount'].max()}, min: {df['Amount'].min()}")

# ==============================================
# Module 4: Class distribution statistics + Pie chart
# ==============================================
# 1. Class distribution statistics
class_count = df['Class'].value_counts()
normal_num = class_count[0]
fraud_num = class_count[1]
fraud_rate = fraud_num / len(df) * 100

print(f"\n===== Class Distribution =====")
print(f"Normal transactions (Class=0): {normal_num}")
print(f"Fraudulent transactions (Class=1): {fraud_num}")
print(f"Fraud rate: {fraud_rate:.2f}%")

# 2. Plot and save pie chart (English labels)
# ********** Replace with your results folder path **********
save_path = 'D:/524/CreditCardFraud_yinyiran/results/class_distribution_pie.png'
plt.figure(figsize=(8, 8))  
labels = [
    'Normal Transactions\n(Class=0)',  
    'Fraudulent Transactions\n(Class=1)'
]
plt.pie(
    [normal_num, fraud_num],
    explode=(0, 0.2),
    labels=labels,
    colors=['#1f77b4', '#ff4b4b'],
    autopct='%1.2f%%',
    shadow=True,
    labeldistance=1.1  
)
plt.title('Credit Card Transaction Class Distribution\n(After Removing Duplicates)', fontsize=14)
plt.tight_layout()
plt.savefig(save_path, bbox_inches='tight')  # 自动裁剪多余空白，保证文字完整
plt.close()
print(f"\nPie chart saved to: {save_path}")

# ==============================================
# Module 5: Save cleaned dataset (for Member 2/3)
# ==============================================
# ********** Replace with your data path **********
clean_data_path = 'D:/524/CreditCardFraud_yinyiran/creditcard_clean.csv'
df.to_csv(clean_data_path, index=False)
print(f"Cleaned dataset saved to: {clean_data_path}")
