import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib

# load tables and create dataframe
df = pd.read_csv(r"F:\1hkm\LU - AIBA\课程\T2\CDS 524 - Machine Learning\Group project\M1_data\creditcard_clean.csv")
explore_table = pd.read_excel(r"F:\1hkm\LU - AIBA\课程\T2\CDS 524 - Machine Learning\Group project\M1_data\dataset_initial_exploration_table.xlsx")

def data_preprocess():
    # data confirm
    print("Core information of explore_table: ")
    print(explore_table)
    print("\nThe basic information of the complete data: ")
    print(f"Total sample size: {len(df)}")
    print(f"Fraudulent samples size: {df['Class'].sum()}")  # Fraud: Class = 1, Non-fraud: Class = 0
    print(f"Feature column：{df.columns.tolist()}")
    print(f"Number of missing values：\n{df.isnull().sum()}")

    # feature_data_splitting
    x = df.drop("Class", axis=1)
    y = df["Class"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

    print(f"Number of training set：{len(x_train)}，Number of test set：{len(x_test)}")
    print(f"Fraud proportion in the training set：{y_train.sum()/len(y_train):.4f}")
    print(f"Fraud proportion in the test set：{y_test.sum()/len(y_test):.4f}")


    # standardscaler
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    X_train_scaled_df = pd.DataFrame(x_train_scaled, columns=x_train.columns)
    X_test_scaled_df = pd.DataFrame(x_test_scaled, columns=x_test.columns)

    # oversampling
    smote = SMOTE(random_state=42)
    x_train_smote, y_train_smote = smote.fit_resample(x_train_scaled, y_train)

    # files_save
    output_path = r"F:\1hkm\LU - AIBA\课程\T2\CDS 524 - Machine Learning\Group project\M3_code\data"

    # Standardized + oversampled training set
    pd.DataFrame(x_train_smote, columns=x.columns).to_csv(output_path + "/x_train_smote.csv", index=False)
    pd.Series(y_train_smote).to_csv(output_path + "/y_train_smote.csv", index=False, header=["Class"])

    # The standardized test set
    pd.DataFrame(x_test_scaled, columns=x.columns).to_csv(output_path + "/x_test_scaled.csv", index=False)
    pd.Series(y_test).to_csv(output_path + "/y_test.csv", index=False, header=["Class"])

    # Standardizer
    joblib.dump(scaler, output_path + "/scaler.pkl")

    # Preprocessing Before and After Comparison Table
    comparison_data = {
        "Data type": ["Original training set", "The training set after oversampling", "Test set"],
        "Total sample size": [len(x_train), len(x_train_smote), len(x_test)],
        "Fraudulent samples size": [y_train.sum(), y_train_smote.sum(), y_test.sum()],
        "Fraud percentage": [f"{y_train.sum() / len(y_train):.4f}", "0.5", f"{y_test.sum() / len(y_test):.4f}"]
    }
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(output_path + "/Comparison Table with Preprocessing Before and After Data.csv", index=False)
    print("The pre-processing files have all been saved.")

data_preprocess()



