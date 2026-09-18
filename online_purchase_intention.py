"""
Session-level online purchase intention modeling.

This script provides a compact, modular baseline workflow based on the
project notebook. Update DATA_PATH to the downloaded UCI CSV before running.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from xgboost import XGBClassifier

DATA_PATH = Path("online_shoppers_intention.csv")
RANDOM_STATE = 42

def load_data(path: Path) -> pd.DataFrame:
    """Load CSV and remove exact duplicate records."""
    df = pd.read_csv(path)
    return df.drop_duplicates().reset_index(drop=True)

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create session-level features using the same input columns for all splits."""
    out = df.copy()
    out["TotalPages"] = out["Administrative"] + out["Informational"] + out["ProductRelated"]
    out["TotalDuration"] = (
        out["Administrative_Duration"] + out["Informational_Duration"]
        + out["ProductRelated_Duration"]
    )
    out["ProductPageShare"] = out["ProductRelated"] / out["TotalPages"].replace(0, np.nan)
    out["ProductDurationShare"] = (
        out["ProductRelated_Duration"] / out["TotalDuration"].replace(0, np.nan)
    )
    out["AvgTimePerProductPage"] = (
        out["ProductRelated_Duration"] / out["ProductRelated"].replace(0, np.nan)
    )
    out["AvgTimePerPage"] = out["TotalDuration"] / out["TotalPages"].replace(0, np.nan)
    # Risk thresholds must be learned from training data; assigned in fit workflow.
    out["HasPageValue"] = (out["PageValues"] > 0).astype(int)
    out["LogPageValues"] = np.log1p(out["PageValues"].clip(lower=0))
    out["IsHolidaySeason"] = out["Month"].isin(["Nov", "Dec"]).astype(int)
    out["NearSpecialDay"] = (out["SpecialDay"] > 0).astype(int)
    out["IsNewVisitor"] = (out["VisitorType"] == "New_Visitor").astype(int)
    return out

def add_training_risk_flags(train: pd.DataFrame, test: pd.DataFrame):
    """Use training-only 75th percentile thresholds for bounce/exit indicators."""
    bounce_cut = train["BounceRates"].quantile(.75)
    exit_cut = train["ExitRates"].quantile(.75)
    for frame in (train, test):
        frame["IsBouncer"] = (frame["BounceRates"] >= bounce_cut).astype(int)
        frame["HighExitRisk"] = (frame["ExitRates"] >= exit_cut).astype(int)
    return train, test

def main():
    df = load_data(DATA_PATH)
    y = df["Revenue"].astype(int)
    X = df.drop(columns=["Revenue"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.20, stratify=y, random_state=RANDOM_STATE
    )
    X_train = engineer_features(X_train)
    X_test = engineer_features(X_test)
    X_train, X_test = add_training_risk_flags(X_train, X_test)

    numeric = X_train.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [c for c in X_train.columns if c not in numeric]
    preprocess = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical)
    ])
    model = XGBClassifier(
        n_estimators=435, max_depth=5, learning_rate=.01,
        min_child_weight=7, subsample=.8, colsample_bytree=.8,
        eval_metric="logloss", random_state=RANDOM_STATE
    )
    pipe = Pipeline([("preprocess", preprocess), ("model", model)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]
    print("Confusion matrix:\n", confusion_matrix(y_test, pred))
    print(classification_report(y_test, pred, digits=4))
    print(f"ROC-AUC: {roc_auc_score(y_test, proba):.4f}")

if __name__ == "__main__":
    main()
