# app.py — Context • Objective • Data • Actions • Observations • Results (Full-data Logistic Regression)
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
DATA_PATH = Path("data/car_insurance.csv")  # adjust if your file lives elsewhere

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
Fit a **Logistic Regression** model on the **full dataset** (no train/test split),
using **all available features** to predict the chosen target. Report overall **Accuracy** and (if applicable) **ROC AUC**,
and list the **most influential features** (coefficients).
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
    miss_pct = (df.isna().mean().mean() * 100.0)
    st.metric("Overall Missing %", f"{miss_pct:.2f}%")

# summary (numeric)
numeric_all = df.select_dtypes(include=[np.number]).columns.tolist()
st.subheader("Summary (Numeric Features)")
if numeric_all:
    desc = df[numeric_all].agg(["count", "mean", "median", "std", "min", "max"]).T
    q = df[numeric_all].quantile([0.25, 0.75])
    desc["q1"] = q.loc[0.25].values
    desc["q3"] = q.loc[0.75].values
    desc["iqr"] = desc["q3"] - desc["q1"]
    desc["missing_%"] = (df[numeric_all].isna().mean() * 100).values
    st.dataframe(
        desc[["count","mean","median","std","min","q1","q3","iqr","max","missing_%"]].round(3),
        use_container_width=True,
    )
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
3. Split features by type (numeric vs categorical).  
4. Preprocess: **StandardScaler** for numeric, **OneHotEncoder(handle_unknown="ignore")** for categorical.  
5. Fit **LogisticRegression(max_iter=1000)** on the **entire dataset** (no train/test).
"""
)
before = len(df)
df_clean = df[~df.duplicated()].copy()
after = len(df_clean)
st.caption(f"Removed {before - after} duplicate rows (kept {after}).")

# ------------------------------------------------------------------------------
# Observations (lightweight)
# ------------------------------------------------------------------------------
st.header("Observations")
obs = []
if numeric_all:
    skew_s = df_clean[numeric_all].skew(numeric_only=True).sort_values(ascending=False)
    right_skewed = [f"{c} (skew={v:.2f})" for c, v in skew_s.head(3).items() if v > 1.0]
    if right_skewed:
        obs.append("Right-skew detected: " + ", ".join(right_skewed) + ".")
for col in numeric_all[:5]:
    series = df_clean[col].dropna()
    if len(series) > 0:
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        obs.append(f"**{col}** typically spans **{q1:,.2f}–{q3:,.2f}** (IQR).")
if obs:
    for o in obs:
        st.markdown(f"- {o}")
else:
    st.markdown("- No notable skew/range signals detected.")

# optional quick EDA plot
if numeric_all:
    st.subheader("Distribution & Box Plot")
    feat = st.selectbox("Choose a numeric feature", options=numeric_all, index=0)
    cA, cB = st.columns(2)
    with cA:
        fig, ax = plt.subplots()
        sns.histplot(df_clean[feat].dropna(), bins=30, ax=ax)
        ax.set_xlabel(feat); ax.set_ylabel("Count")
        st.pyplot(fig)
    with cB:
        fig, ax = plt.subplots()
        sns.boxplot(x=df_clean[feat].dropna(), ax=ax)
        ax.set_xlabel(feat); ax.set_ylabel("")
        st.pyplot(fig)

# ------------------------------------------------------------------------------
# Results — Full-dataset Logistic Regression (no train/test, no confusion matrix)
# ------------------------------------------------------------------------------
st.header("Results — Logistic Regression (Full Data)")

# choose target (required)
common_targets = ["target","label","claim","claim_status","made_claim","fraud_flag","is_fraud","fraud_found"]
guesses = [c for c in df_clean.columns if c.lower() in common_targets]
default_idx = (df_clean.columns.get_loc(guesses[0]) if guesses else 0)
target_col = st.selectbox("Select target column", options=df_clean.columns, index=default_idx)

if target_col not in df_clean.columns:
    st.error("Please select a valid target column.")
    st.stop()

# features/target
y = df_clean[target_col]
X = df_clean.drop(columns=[target_col])

# types
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]

# preprocessing + model
pre = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ],
    remainder="drop",
)

pipe = Pipeline(steps=[("preprocess", pre), ("clf", LogisticRegression(max_iter=1000))])

# fit on the *entire* dataset
pipe.fit(X, y)

# in-sample predictions & metrics
y_pred = pipe.predict(X)
acc = accuracy_score(y, y_pred)

st.subheader("Metrics (In-sample)")
st.write(f"**Accuracy:** {acc:.3f}")

# AUC only if binary & proba available
try:
    if y.nunique() == 2 and hasattr(pipe.named_steps["clf"], "predict_proba"):
        y_prob = pipe.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, y_prob)
        st.write(f"**ROC AUC:** {auc:.3f}")
except Exception:
    pass

# ----------------- Feature importance (coefficients) -----------------
st.subheader("Top Features (Coefficients)")

# recover feature names from ColumnTransformer
def get_feature_names(ct: ColumnTransformer, numeric_names, categorical_names):
    feature_names = []
    for name, trans, cols in ct.transformers_:
        if name == "num":
            # scaler doesn't change names
            feature_names.extend(list(cols))
        elif name == "cat":
            # onehot supplies get_feature_names_out
            ohe = trans
            try:
                ohe_names = ohe.get_feature_names_out(cols)
            except Exception:
                # sklearn < 1.0 fallback
                ohe_names = ohe.get_feature_names(cols)
            feature_names.extend(list(ohe_names))
        # ignore remainder
    return feature_names

feat_names = get_feature_names(pipe.named_steps["preprocess"], num_cols, cat_cols)

# coefficients
coef = pipe.named_steps["clf"].coef_
if coef.ndim == 1:
    coef = coef.reshape(1, -1)

# For binary classification, coef shape = (1, n_features)
coef_vec = coef[0]
coef_df = pd.DataFrame({"feature": feat_names, "coefficient": coef_vec})
coef_df["abs_coeff"] = coef_df["coefficient"].abs()
coef_df = coef_df.sort_values("abs_coeff", ascending=False)

top_n = st.slider("Show top N features", 5, min(25, len(coef_df)), min(10, len(coef_df)))
st.dataframe(coef_df.head(top_n)[["feature", "coefficient"]].round(4), use_container_width=True)

# simple horizontal bar for top coefficients
fig, ax = plt.subplots(figsize=(7, max(3, int(top_n*0.4))))
plot_df = coef_df.head(top_n).sort_values("coefficient")
ax.barh(plot_df["feature"], plot_df["coefficient"])
ax.set_xlabel("Coefficient (Logistic Regression)")
ax.set_ylabel("")
st.pyplot(fig)

# ------------------------------------------------------------------------------
# Footer
# ------------------------------------------------------------------------------
st.markdown("---")
st.caption("Model fits on the full dataset (resubstitution). Add validation if you need out-of-sample estimates.")
