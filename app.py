# app.py — Context • Objective • Data • Actions • Observations • Results (Best Single Feature Accuracy)
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

# Viz & ML
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    f1_score,
)

# ------------------------------------------------------------------------------
# Page setup
# ------------------------------------------------------------------------------
st.set_page_config(page_title="Project Report", page_icon="📑", layout="wide")
st.title("📑 Project Report")
st.caption("Context • Objective • About the Data • Actions • Observations • Results")

# ------------------------------------------------------------------------------
# Load data (fixed path with fallback to uploader)
# ------------------------------------------------------------------------------
DATA_PATH = Path("data/car_insurance.csv")  # change if your file lives elsewhere

@st.cache_data(show_spinner=False)
def read_csv_forgiving(src):
    """Try common CSV read variants to survive odd delimiters/encodings."""
    try:
        return pd.read_csv(src)
    except Exception:
        try:
            return pd.read_csv(src, engine="python")
        except Exception:
            return pd.read_csv(src, sep=";", engine="python")

def load_data():
    if DATA_PATH.exists():
        df = read_csv_forgiving(DATA_PATH)
        src = f"`{DATA_PATH}`"
    else:
        up = st.file_uploader("Upload a CSV", type=["csv"])
        if not up:
            st.info("Add a dataset at `data/...csv` or upload a CSV to proceed.")
            st.stop()
        df = read_csv_forgiving(up)
        src = "uploaded file"
    # normalize headers
    df.columns = df.columns.str.strip().str.replace(r"\s+", "_", regex=True)
    return df, src

df, src = load_data()
st.success(f"Dataset loaded from {src}")

# ------------------------------------------------------------------------------
# SECTION: Context
# ------------------------------------------------------------------------------
st.header("Context")
st.markdown(
    """
Organizations need clear visibility into customer behaviors and drivers to make targeted, high-impact decisions.
This project turns raw data into structured insights that can inform marketing, risk, and operations.
"""
)

# ------------------------------------------------------------------------------
# SECTION: Objective
# ------------------------------------------------------------------------------
st.header("Objective")
st.markdown(
    """
Select the **single most predictive feature** for a chosen target by comparing
**Accuracy** (primary), with **F1** and **ROC AUC** as additional context (when applicable).
"""
)

# ------------------------------------------------------------------------------
# SECTION: About the Data
# ------------------------------------------------------------------------------
st.header("About the Data")
c0, c1 = st.columns([2, 1])
with c0:
    st.subheader("Preview")
    st.dataframe(df.head(), use_container_width=True)

with c1:
    st.subheader("Shape & Missingness")
    st.metric("Rows", len(df))
    st.metric("Columns", df.shape[1])
    miss_pct = (df.isna().mean().mean() * 100.0)
    st.metric("Overall Missing %", f"{miss_pct:.2f}%")

# Summary table (numeric only)
numeric_cols_all = df.select_dtypes(include=[np.number]).columns.tolist()
st.subheader("Summary (Numeric Features)")
if numeric_cols_all:
    desc = df[numeric_cols_all].agg(["count", "mean", "median", "std", "min", "max"]).T
    q = df[numeric_cols_all].quantile([0.25, 0.75])
    desc["q1"] = q.loc[0.25].values
    desc["q3"] = q.loc[0.75].values
    desc["iqr"] = desc["q3"] - desc["q1"]
    desc["missing_%"] = (df[numeric_cols_all].isna().mean() * 100).values
    st.dataframe(
        desc[
            ["count", "mean", "median", "std", "min", "q1", "q3", "iqr", "max", "missing_%"]
        ].round(3),
        use_container_width=True,
    )
else:
    st.info("No numeric columns found.")

# ------------------------------------------------------------------------------
# SECTION: Actions — Data Preparation
# ------------------------------------------------------------------------------
st.header("Actions — Data Preparation")
st.markdown(
    """
**Steps performed**
1. **Header normalization** (trim spaces; replace internal spaces with underscores).  
2. **Duplicate removal** (exact duplicate rows).  
3. **Type inspection**; numeric/categorical split for modeling.  
4. **Per-feature pipelines**:  
   - Numeric → `StandardScaler`  
   - Categorical → `OneHotEncoder(handle_unknown="ignore")`  
   - Estimator → `LogisticRegression(max_iter=1000)`  
"""
)

before = len(df)
df_clean = df[~df.duplicated()].copy()
after = len(df_clean)
st.caption(f"Removed {before - after} duplicate rows (kept {after}).")

# ------------------------------------------------------------------------------
# SECTION: Observations (light, data-driven)
# ------------------------------------------------------------------------------
st.header("Observations")
obs = []
# skew signal
if numeric_cols_all:
    skew_s = df_clean[numeric_cols_all].skew(numeric_only=True).sort_values(ascending=False)
    right_skewed = [f"{c} (skew={v:.2f})" for c, v in skew_s.head(3).items() if v > 1.0]
    if right_skewed:
        obs.append("Right-skew detected: " + ", ".join(right_skewed) + ".")
# typical ranges
for col in numeric_cols_all[:5]:
    series = df_clean[col].dropna()
    if len(series) > 0:
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        obs.append(f"**{col}** typically spans **{q1:,.2f}–{q3:,.2f}** (IQR).")
if obs:
    for o in obs:
        st.markdown(f"- {o}")
else:
    st.markdown("- No notable skew/range signals detected.")

# Quick EDA plot
if numeric_cols_all:
    st.subheader("Distribution & Box Plot")
    feat_for_plot = st.selectbox("Choose a numeric feature", options=numeric_cols_all, index=0, key="eda_feat")
    cA, cB = st.columns(2)
    with cA:
        fig, ax = plt.subplots()
        sns.histplot(df_clean[feat_for_plot].dropna(), bins=30, ax=ax)
        ax.set_xlabel(feat_for_plot); ax.set_ylabel("Count")
        st.pyplot(fig)
    with cB:
        fig, ax = plt.subplots()
        sns.boxplot(x=df_clean[feat_for_plot].dropna(), ax=ax)
        ax.set_xlabel(feat_for_plot); ax.set_ylabel("")
        st.pyplot(fig)

# ------------------------------------------------------------------------------
# SECTION: Results — Best Single Feature by Accuracy (no clustering)
# ------------------------------------------------------------------------------
st.header("Results — Best Single Feature by Accuracy")

# Target selection (required)
common_targets = ["target", "label", "claim", "claim_status", "made_claim", "fraud_flag", "is_fraud", "fraud_found"]
guesses = [c for c in df_clean.columns if c.lower() in common_targets]
default_idx = (df_clean.columns.get_loc(guesses[0]) if guesses else 0)
target_col = st.selectbox("Select target column (binary preferred)", options=df_clean.columns, index=default_idx)

if target_col not in df_clean.columns:
    st.error("Please select a valid target column.")
    st.stop()

y = df_clean[target_col]
X_all = df_clean.drop(columns=[target_col])

# Identify types
num_cols = X_all.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in X_all.columns if c not in num_cols]

# Train/validation split (stratify for classification if sensible)
stratify_arg = y if y.nunique() <= 20 else None
test_size = st.slider("Test size", 0.10, 0.40, 0.20, 0.05)
random_state = 42

# Helper to evaluate a single feature
def eval_single_feature(feature_name: str):
    X = df_clean[[feature_name]]
    # Determine type
    if feature_name in num_cols:
        pre = ColumnTransformer(
            transformers=[("num", StandardScaler(), [feature_name])],
            remainder="drop",
        )
    else:
        pre = ColumnTransformer(
            transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), [feature_name])],
            remainder="drop",
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_arg
    )

    model = Pipeline(steps=[("preprocess", pre), ("clf", LogisticRegression(max_iter=1000))])
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    # F1 (binary or multi)
    try:
        average = "binary" if y_test.nunique() == 2 else "weighted"
        f1 = f1_score(y_test, y_pred, average=average, zero_division=0)
    except Exception:
        f1 = np.nan

    # AUC (only meaningful for binary and if predict_proba exists)
    auc = np.nan
    try:
        if hasattr(model.named_steps["clf"], "predict_proba") and y_test.nunique() == 2:
            y_prob = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_prob)
    except Exception:
        pass

    return {
        "feature": feature_name,
        "type": "numeric" if feature_name in num_cols else "categorical",
        "accuracy": acc,
        "f1": f1,
        "auc": auc,
    }

# Evaluate all features one-by-one
results = []
for feat in X_all.columns:
    try:
        results.append(eval_single_feature(feat))
    except Exception as e:
        # Keep going even if a single feature fails (e.g., constant col)
        results.append({"feature": feat, "type": "error", "accuracy": np.nan, "f1": np.nan, "auc": np.nan})

res_df = pd.DataFrame(results)

# Rank by accuracy, then by AUC as tie-breaker
res_df["accuracy_rank"] = (-res_df["accuracy"].fillna(-1), -res_df["auc"].fillna(-1))
res_df = res_df.sort_values(by=["accuracy", "auc"], ascending=[False, False]).drop(columns=["accuracy_rank"])

# Display best feature
if not res_df.empty and res_df["accuracy"].notna().any():
    best_row = res_df.iloc[0]
    st.subheader("Best Single Feature")
    st.markdown(
        f"- **Feature:** `{best_row['feature']}`  \n"
        f"- **Type:** {best_row['type']}  \n"
        f"- **Accuracy:** **{best_row['accuracy']:.3f}**  \n"
        f"- **F1:** {'' if np.isnan(best_row['f1']) else f'{best_row['f1']:.3f}'}  \n"
        f"- **ROC AUC:** {'' if np.isnan(best_row['auc']) else f'{best_row['auc']:.3f}'}"
    )
else:
    st.info("Could not compute feature-wise metrics. Check the target and feature columns.")

# Show full ranking table (styled)
st.subheader("Feature Ranking by Accuracy")
disp = res_df.copy().reset_index(drop=True).round({"accuracy": 3, "f1": 3, "auc": 3})
styler = (
    disp.style
    .bar(subset=["accuracy"], color="#66b3ff")
    .background_gradient(subset=["auc"], cmap="Greens")
    .hide(axis="index")
)
st.dataframe(styler, use_container_width=True)

# Optional: confusion matrix for best feature model
if not res_df.empty and res_df["accuracy"].notna().any():
    # retrain best single-feature model to show CM
    best_feat = best_row["feature"]
    st.subheader(f"Confusion Matrix — Best Feature: {best_feat}")

    Xb = df_clean[[best_feat]]

    if best_feat in num_cols:
        preb = ColumnTransformer(
            transformers=[("num", StandardScaler(), [best_feat])],
            remainder="drop",
        )
    else:
        preb = ColumnTransformer(
            transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), [best_feat])],
            remainder="drop",
        )

    X_train, X_test, y_train, y_test = train_test_split(
        Xb, y, test_size=test_size, random_state=42, stratify=stratify_arg
    )
    modelb = Pipeline(steps=[("preprocess", preb), ("clf", LogisticRegression(max_iter=1000))])
    modelb.fit(X_train, y_train)
    y_pred_b = modelb.predict(X_test)

    cm = confusion_matrix(y_test, y_pred_b)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cbar=False, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(f"Confusion Matrix ({best_feat})")
    st.pyplot(fig)

# ------------------------------------------------------------------------------
# Footer
# ------------------------------------------------------------------------------
st.markdown("---")
st.caption("This report identifies the best single predictive feature by Accuracy (with F1/AUC context).")
