# app.py — Context • Objective • Data • Actions • Observations • Results
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
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.cluster import KMeans

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
Identify patterns and segments that explain **capacity** (e.g., limits, product penetration) and **behavior**
(e.g., channel usage), and—when a labeled target is available—build a **baseline predictive model** to support decisions.
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
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
st.subheader("Summary (Numeric Features)")
if numeric_cols:
    desc = df[numeric_cols].agg(["count", "mean", "median", "std", "min", "max"]).T
    q = df[numeric_cols].quantile([0.25, 0.75])
    desc["q1"] = q.loc[0.25].values
    desc["q3"] = q.loc[0.75].values
    desc["iqr"] = desc["q3"] - desc["q1"]
    desc["missing_%"] = (df[numeric_cols].isna().mean() * 100).values
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
3. **Type inspection**; **numeric subset** prepared for EDA and modeling.  
4. For modeling: **Standardize numeric** and **One-Hot encode categorical** features via a `ColumnTransformer`.
"""
)

# Apply duplicate removal (keep original index for potential alignment if needed)
before = len(df)
df_clean = df[~df.duplicated()].copy()
after = len(df_clean)

st.caption(f"Removed {before - after} duplicate rows (kept {after}).")

# ------------------------------------------------------------------------------
# SECTION: Observations (auto-generated from current data)
# ------------------------------------------------------------------------------
st.header("Observations")
obs = []

# 1) Skew cues (top 3 right-skewed)
if numeric_cols:
    skew_s = df_clean[numeric_cols].skew(numeric_only=True).sort_values(ascending=False)
    right_skewed = [f"{c} (skew={v:.2f})" for c, v in skew_s.head(3).items() if v > 1.0]
    if right_skewed:
        obs.append("Right-skew detected: " + ", ".join(right_skewed) + ".")

# 2) Typical ranges (IQR)
for col in numeric_cols[:5]:  # keep it concise
    series = df_clean[col].dropna()
    if len(series) > 0:
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        obs.append(f"**{col}** typically spans **{q1:,.2f}–{q3:,.2f}** (IQR).")

# 3) Correlations (top absolute)
if len(numeric_cols) >= 2:
    corr_abs = df_clean[numeric_cols].corr(numeric_only=True).abs()
    upper = corr_abs.where(np.triu(np.ones(corr_abs.shape), k=1).astype(bool))
    top_corr = upper.stack().sort_values(ascending=False).head(5)
    if not top_corr.empty:
        txt = "; ".join([f"{a}–{b} ({v:.2f})" for (a, b), v in top_corr.items()])
        obs.append(f"Strongest numeric correlations: {txt}.")

if obs:
    for o in obs:
        st.markdown(f"- {o}")
else:
    st.markdown("- No notable skew or correlation signals detected in the current dataset.")

# Optional quick EDA plot
if numeric_cols:
    st.subheader("Distribution & Box Plot")
    feat = st.selectbox("Choose a numeric feature", options=numeric_cols, index=0)
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
# SECTION: Results
# - If a binary target exists (or selected), show baseline classifier results.
# - Otherwise, run KMeans (K=3) on numeric features and show cluster profiles.
# ------------------------------------------------------------------------------
st.header("Results")

# Try to guess a binary-ish target; let user override
candidates = [c for c in df_clean.columns if c.lower() in
              ("target", "label", "fraud_found", "is_fraud", "fraud_flag", "claim", "claim_status", "made_claim")]
default_idx = (df_clean.columns.get_loc(candidates[0]) if candidates else 0)
target_col = st.selectbox(
    "Target (optional; leave as-is if unsupervised)",
    options=["(no target)"] + df_clean.columns.tolist(),
    index=0 if not candidates else (df_clean.columns.get_loc(candidates[0]) + 1)
)

supervised = target_col != "(no target)"

if supervised:
    y = df_clean[target_col]
    X = df_clean.drop(columns=[target_col])

    num_cols_model = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols_model = [c for c in X.columns if c not in num_cols_model]

    st.markdown("**Baseline Classifier: Logistic Regression**")
    st.caption("Numeric → StandardScaler; Categorical → OneHotEncoder")

    stratify_arg = y if y.nunique() <= 20 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify_arg
    )

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols_model),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols_model),
        ],
        remainder="drop",
    )

    pipe = Pipeline(steps=[("preprocess", pre), ("clf", LogisticRegression(max_iter=1000))])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    st.subheader("Classification Report")
    rep = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True, zero_division=0)).T.round(3)
    st.dataframe(rep, use_container_width=True)

    # ROC AUC if binary probabilities available
    try:
        if y_test.nunique() == 2 and hasattr(pipe.named_steps["clf"], "predict_proba"):
            y_prob = pipe.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_prob)
            st.write(f"**ROC AUC:** {auc:.3f}")
    except Exception:
        pass

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cbar=False, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion Matrix")
    st.pyplot(fig)

else:
    # Unsupervised: KMeans (K=3) on numeric columns
    if len(numeric_cols) < 2:
        st.info("Not enough numeric features to run clustering.")
    else:
        st.markdown("**Unsupervised Segmentation: KMeans (K=3)**")
        X = df_clean[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).values
        X_scaled = StandardScaler().fit_transform(X)

        k = 3
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X_scaled)

        scored = df_clean.copy()
        scored["CLUSTER"] = labels

        st.subheader("Cluster Profiles (Mean by Feature)")
        prof = scored.groupby("CLUSTER")[numeric_cols].mean().round(2)
        st.dataframe(prof, use_container_width=True)

        # Simple cluster descriptions (relative z across clusters)
        st.subheader("Cluster Descriptions")
        prof_z = (prof - prof.mean()) / (prof.std(ddof=0).replace(0, 1))
        for cid, row in prof_z.iterrows():
            top_feats = ", ".join(row.sort_values(ascending=False).head(3).index.tolist())
            st.markdown(f"- **Cluster {cid}** — higher on: {top_feats}")

        # Download labeled data
        st.download_button(
            "Download data with cluster labels (CSV)",
            data=scored.to_csv(index=False),
            file_name="clustered_output.csv",
            mime="text/csv",
        )

# ------------------------------------------------------------------------------
# Footer
# ------------------------------------------------------------------------------
st.markdown("---")
st.caption("Generated with Streamlit • Edit the text in each section to match your domain narrative.")
