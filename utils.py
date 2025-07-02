import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ------------------
# Core data helpers
# ------------------
CATEGORICAL_COLS = [
    'Age_Group','Gender','Department','Job_Level','Tenure','Work_Arrangement',
    'Employment_Type','Training_Recency','Role_Specific_Training',
    'Password_Change_Freq','MFA_Use','Personal_Device_Use',
    'Report_Suspicious_Email','Admin_Privileges',
    'Incidents_Caused_Category'
]

TARGET_CLASS = 'Phishing_Pass'
TARGET_REG   = 'Phishing_Score'

def load_data(path, uploaded_file=None):
    """Return df from uploaded file or default path."""
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_csv(path)
    return df

# ------------------
# Pre‑processing
# ------------------
def preprocess_for_classification(df):
    """Return X,y after minimal preprocessing (one‑hot for categoricals)."""
    X = df.drop(columns=[TARGET_CLASS])
    y = df[TARGET_CLASS].map({'Yes':1,'No':0})
    X_enc = pd.get_dummies(X, columns=CATEGORICAL_COLS, drop_first=True)
    return X_enc, y

def preprocess_for_regression(df):
    """Return X,y for regression w/out target cols."""
    X = df.drop(columns=[TARGET_REG])
    y = df[TARGET_REG]
    X_enc = pd.get_dummies(X, columns=CATEGORICAL_COLS + [TARGET_CLASS], drop_first=True)
    return X_enc, y

# ------------------
# Generic metrics
# ------------------
def compute_clf_metrics(y_true, y_pred):
    return dict(
        accuracy=accuracy_score(y_true, y_pred),
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        f1=f1_score(y_true, y_pred, zero_division=0)
    )