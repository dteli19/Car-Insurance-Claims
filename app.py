# app.py — Context • Objective • Data • Actions • Observations • Results (Auto best single feature by accuracy)
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

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
# Context
# ------------------------------------------------------------------------------
st.header("Context")
st.markdown(
    """
Organizations need clear visibility into customer behaviors and drivers to make targeted, high-impact decisions.
This project turns raw data into structured insights that can inform marketing, risk, and operations.
"""
)

# ------------------------------------------------------------------------------
# Objective
# ------------------------------------------------------------------------------
st.header("Objective")
st.markdown(
    """
Identify the **single most predictive feature** for a **binary target** using **Logistic Regression** on the **full dataset**
(no train/test split). Report **Accuracy** (primary) and **ROC AUC** when applicable.
"""
)

# ------------------------------------------------------------------------------
# About the Data
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
    st.metric("Overall Missing %", f"{(df.isna().mean().mean()*100):.2f}%")

numeric_all = df.select_dtypes(include=[np.number]).columns.tolist()
st.subheader("Summary (Numeric Features)")
if numeric_all:
    desc = df[numeric_all].agg(["count","mean","median","std","min","max"]).T
    q = df[numeric_all].quantile([0.25,0.75])
    desc["q1"] = q.loc[0.25].values
    desc["q3"] = q.loc[0.75].values
    desc["iqr"] = desc["q3"] - desc["q1"]
    desc["missing_%"] = (df[numeric_all].isna().mean()*100).values
    st.dataframe(desc[["count","mean","median","std","min","q1","q3","iqr","max","missing_%"]].round(3),
                 use_container_width=True)
else:
    st.info("No numeric columns found.")

# ------------------------------------------------------------------------------
# Actions — Data Preparation
# ------------------------------------------------------------------------------
st.header("Actions — Data Preparation")
st.markdown(
    """
**Steps performed**
1. Header normalization (trim, underscores).  
2. Drop exact duplicate rows.  
3. Identify numeric vs categorical features.  
4. For each candidate single feature:  
   - Numeric → `StandardScaler`  
   - Categorical → `OneHotEncoder(handle_unknown="ignore")`  
   - Model → `LogisticRegression(max_iter=1000)` fit on **full data** (resubstitution).
"""
)
before = len(df)
df_clean = df[~df.uplicated()].copy() if hasattr(df, "uplicated") else df[~df.duplicated()].copy()
after = len(df_clean)
st.caption(f"Removed {before - after} duplicate rows (kept {after}).")

# ------------------------------------------------------------------------------
# Observations (light)
# ------------------------------------------------------------------------------
st.header("Observations")
obs = []
if numeric_all:
    skew_s = df_clean[numeric_all].skew(numeric_only=True).sort_values(ascending=False)
    right_skewed = [f"{c} (skew={v:.2f})" for c, v in skew_s.head(3).items() if v > 1.0]
    if right_skewed:
        obs.append("Right-skew detected: " + ", ".join(right_skewed) + ".")
for col in numeric_all[:5]:
    s = df_clean[col].dropna()
    if len(s) > 0:
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        obs.append(f"**{col}** typically spans **{q1:,.2f}–{q3:,.2f}** (IQR).")
for o in (obs or ["- No notable skew/range signals detected."]):
    st.markdown(f"- {o}")

# ------------------------------------------------------------------------------
# Results — Auto best feature by Accuracy (no UI selections, no split)
# ------------------------------------------------------------------------------
st.header("Results — Best Single Feature (Accuracy)")

# 1) Auto-detect a binary target
common_targets = ["target","label","claim","claim_status","made_claim","fraud_flag","is_fraud","fraud_found"]
binary_cols = [c for c in df_clean.columns if df_clean[c].dropna().nunique() == 2]

target_col = None
# Prefer common names that are binary
for c in df_clean.columns:
    if c.lower() in common_targets and c in binary_cols:
        target_col = c
        break
# Otherwise pick the first binary column (not obviously an ID)
if target_col is None and binary_cols:
    # deprioritize columns that look like identifiers
    id_like = {"id","sl_no","slno","customer_key","customerid","customer_id"}
    sorted_bins = sorted(binary_cols, key=lambda x: (x.lower() in id_like, df_clean.columns.get_loc(x)))
    target_col = sorted_bins[0]

if target_col is None:
    st.error("No binary target column found. Please include a binary target (e.g., claim, fraud_flag).")
    st.stop()

st.markdown(f"**Detected target:** `{target_col}` (binary)")

y = df_clean[target_col].dropna()
X_full = df_clean.loc[y.index].drop(columns=[target_col])

# Remove columns that are entirely missing or constant
def is_constant(s: pd.Series) -> bool:
    s_non_null = s.dropna()
    return s_non_null.nunique() <= 1

valid_features = [c for c in X_full.columns if not is_constant(X_full[c])]

if not valid_features:
    st.error("No valid feature columns available after cleaning.")
    st.stop()

# Split types
num_cols = X_full[valid_features].select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in valid_features if c not in num_cols]

# 2) Evaluate each feature (fit on full data; predict on full data)
results = []
for feat in valid_features:
    Xi = X_full[[feat]]
    if feat in num_cols:
        pre = ColumnTransformer([("num", StandardScaler(), [feat])], remainder="drop")
    else:
        pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), [feat])], remainder="drop")
    pipe = Pipeline([("preprocess", pre), ("clf", LogisticRegression(max_iter=1000))])

    try:
        pipe.fit(Xi, y)
        y_pred = pipe.predict(Xi)
        acc = accuracy_score(y, y_pred)
        auc = np.nan
        # AUC only if proba exists (binary)
        try:
            y_prob = pipe.predict_proba(Xi)[:, 1]
            auc = roc_auc_score(y, y_prob)
        except Exception:
            pass
        results.append({"feature": feat, "type": "numeric" if feat in num_cols else "categorical",
                        "accuracy": acc, "auc": auc})
    except Exception:
        results.append({"feature": feat, "type": "error", "accuracy": np.nan, "auc": np.nan})

res_df = pd.DataFrame(results)
if res_df.empty or res_df["accuracy"].isna().all():
    st.error("Unable to compute accuracy for any feature. Check data types and target.")
    st.stop()

# Rank by accuracy, tie-breaker by AUC
res_df = res_df.sort_values(by=["accuracy", "auc"], ascending=[False, False]).reset_index(drop=True)

best = res_df.iloc[0]
st.subheader("Best Feature")
st.markdown(
    f"- **Feature:** `{best['feature']}`  \n"
    f"- **Type:** {best['type']}  \n"
    f"- **Accuracy:** **{best['accuracy']:.3f}**  \n"
    f"- **ROC AUC:** {'' if np.isnan(best['auc']) else f'{best['auc']:.3f}'}"
)

st.subheader("All Features — Ranked by Accuracy")
disp = res_df.copy().round({"accuracy": 3, "auc": 3})
styler = (
    disp.style
    .bar(subset=["accuracy"], color="#76b7ff")
    .background_gradient(subset=["auc"], cmap="Greens")
    .hide(axis="index")
)
st.dataframe(styler, use_container_width=True)

# ------------------------------------------------------------------------------
# Footer
# ------------------------------------------------------------------------------
st.markdown("---")
st.caption("Logistic Regression on full data (resubstitution). No user selections; best single feature auto-identified by Accuracy.")
