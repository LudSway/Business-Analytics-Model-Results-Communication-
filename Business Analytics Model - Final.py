#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.impute import KNNImputer
from sklearn.metrics import (accuracy_score, confusion_matrix, roc_auc_score, roc_curve, 
                             classification_report, precision_recall_curve, 
                             precision_score, recall_score, balanced_accuracy_score)
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import uniform, randint

# Load the dataset
try:
    file_path = r'C:\Users\dell\OneDrive\Desktop\Cleaned_Dataset_Final_Filled.xlsx'  
    df = pd.read_excel(file_path)
except PermissionError as e:
    print(f"Permission Error: {e}")
    print("Please make sure the file is not open in another application and check the file path.")
except FileNotFoundError as e:
    print(f"File Not Found Error: {e}")
    print("Please check if the file path is correct.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

# Conducting EDA on the file
if 'df' in locals():
    print("Basic Information:")
    print(df.info())
    print("\nFirst few rows of the dataset:")
    print(df.head())

    # Check for missing values
    print("\nMissing values in each column:")
    missing_values = df.isnull().sum()
    print(missing_values)

    # Handle missing values
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    string_columns = df.select_dtypes(include=[object]).columns

    # Filling missing numeric values with KNNImputer
    imputer = KNNImputer(n_neighbors=3)
    df[numeric_columns] = imputer.fit_transform(df[numeric_columns])

    # Filling missing string values with 'Unknown'
    df[string_columns] = df[string_columns].fillna('Unknown')

    # Creating 'Churn' column based on customer activity
    churn_threshold = pd.to_datetime('2024-03-01')  # Adjust the date as needed
    df['TransactionDate'] = pd.to_datetime(df['TransactionDate'])  # Ensure datetime format
    df['Churn'] = np.where(df['TransactionDate'] < churn_threshold, 1, 0)  # Churn label

    target_column = 'Churn'

    # Checking class distribution in 'Churn'
    print(f"\nClass Distribution in {target_column}:")
    print(df[target_column].value_counts())

    # For large datasets, consider taking a subset for faster experimentation
    df_numeric = df.select_dtypes(include=[np.number])
    X = df_numeric.drop(columns=[target_column])
    y = df_numeric[target_column]

    # Feature selection
    selector = SelectKBest(f_classif, k=3)  # Adjust 'k' based on your requirement
    X_selected = selector.fit_transform(X, y)
    
    # Handling imbalanced dataset using SMOTEENN (Apply SMOTE and Edited Nearest Neighbors)
    smote_enn = SMOTEENN(random_state=42)  # Combine SMOTE and ENN to handle oversampling and clean noise
    X_resampled, y_resampled = smote_enn.fit_resample(X_selected, y)

    # Check resampled class distribution
    print(f"\nResampled class distribution using SMOTEENN:\n{y_resampled.value_counts()}")

    # Splitting data into train and test sets (stratified)
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

    # Scaling data
    scaler = StandardScaler()
    X_train_stzd = scaler.fit_transform(X_train)
    X_test_stzd = scaler.transform(X_test)

    # Logistic Regression with class_weight='balanced'
    lr_model = LogisticRegression(class_weight='balanced', solver='liblinear', random_state=42)
    lr_model.fit(X_train_stzd, y_train)

    # Random Forest with class_weight='balanced'
    rf_model = RandomForestClassifier(class_weight='balanced', random_state=42)
    rf_model.fit(X_train_stzd, y_train)

    # XGBoost with scale_pos_weight to handle class imbalance
    xgb_model = XGBClassifier(scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(), 
                              use_label_encoder=False, eval_metric='logloss', random_state=42)
    xgb_model.fit(X_train_stzd, y_train)

    # Evaluate all models
    models = {
        'Logistic Regression': lr_model,
        'Random Forest': rf_model,
        'XGBoost': xgb_model
    }

    for name, model in models.items():
        y_pred = model.predict(X_test_stzd)
        y_pred_prob = model.predict_proba(X_test_stzd)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_prob)
        balanced_acc = balanced_accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        print(f"\n{name} Model Evaluation:")
        print(f"Accuracy: {accuracy}")
        print(f"Balanced Accuracy: {balanced_acc}")
        print(f"ROC-AUC Score: {roc_auc}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))

        # Precision-Recall Curve
        precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_prob)
        plt.plot(recall_vals, precision_vals, label=f'Precision-Recall Curve ({name})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve ({name})')
        plt.legend(loc='lower left')
        plt.show()

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve ({name})')
        plt.legend(loc='lower right')
        plt.show()

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix ({name})')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()


# In[ ]:




