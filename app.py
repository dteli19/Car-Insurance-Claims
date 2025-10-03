# app.py — Car Insurance | Claims Risk Classifier (Streamlit)
import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Viz & ML
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="Car Insurance — Claims Risk Classifier", page_icon="🚗", layout="wide")
st.title("🚗 Car Insurance — Claims Risk Classifier")
st.caption("Context → Objective → Data → Actions → Results")

# Optional banner
banner = Path("assets/car.jpg")
if banner.exists():
    st.image(str(banner), use_container_width=True)

with st.expander("Context & Objective", expanded=True):
    st.markdown(
        """
**Context.** Insurers need to flag risky claims early (e.g., fraud/high loss) without slowing straight-through processing.

**Objective.** Build a lightweight, reproducible **classification app** that:
- explores the car-insurance dataset (EDA),
- trains a **baseline model** (Logistic Regression) to predict a binary **target**,
- **scores** new CSVs for quick experimentation.
"""
    )

# -----------------------------
# Data loader (fixed file or uploader)
# -----------------------------
DATA_PATH = Path("data/car_insurance.csv")

@st.cache_data(show_spinner=False)
def _read_csv_any(path_or_buffer):
    """
    More forgiving CSV reader:
    - tries pandas defaults
    - then tries Python engine & ; delimiter
    """
    try:
        return pd.read_csv(path_or_buffer)
    except Exception:
        try:
            return pd.read_csv(path_or_buffer, engine="python")
        except Exception:
            return pd.read_csv(path_or_buffer, sep=";", engine="python")

def load_data():
    if DATA_PATH.exists():
        df = _read_csv_any(DATA_PATH)
        src = f"`{DATA_PATH}`"
    else:
        uploaded = st.file_uploader("Upload a CSV (features + target)", type=["csv"])
        if not uploaded:
            st.info("Add `data/car_insurance.csv` to the repo or upload a CSV.")
            st.stop()
        df = _read_csv_any(uploaded)
        src = "uploaded file"
    # normalize headers: strip spaces and replace internal spaces with underscores
    df.columns = df.columns.str.strip().str.replace(r"\s+", "_", regex=True)
    return df, src

df, src = load_data()
st.success(f"Loaded dataset from {src} ✅")
st.subheader("Data Preview")
st.dataframe(df.head(), use_container_width=True)

# -----------------------------
# Target selection & schema
# -----------------------------
DEFAULT_TARGETS = [
    # common names seen in Colab notebooks
    "fraud_found", "is_fraud", "fraud_flag",
    "claim", "claim_status", "made_claim", "target", "label"
]
guesses = [c for c in df.columns if c.lower() in DEFAULT_TARGETS]
default_idx = (df.columns.get_loc(guesses[0]) if guesses else 0)

target = st.selectbox("Select target column (prefer binary 0/1)", options=df.columns, index=default_idx)

if target not in df.columns:
    st.error("Select a valid target column.")
    st.stop()

X = df.drop(columns=[target])
y = df[target]

# Identify numeric / categorical columns
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]

with st.expander("Schema", expanded=False):
    st.write("**Numeric features:**", num_cols if num_cols else "(none)")
    st.write("**Categorical features:**", cat_cols if cat_cols else "(none)")
    st.write("**Target:**", target)

# -----------------------------
# Quick EDA
# -----------------------------
st.markdown("### Exploratory Analysis")
feat = st.selectbox("Select a feature to explore", options=X.columns)

c1, c2 = st.columns(2)

with c1:
    st.subheader("Histogram / Bar")
    fig, ax = plt.subplots()
    if pd.api.types.is_numeric_dtype(X[feat]):
        sns.histplot(X[feat].dropna(), bins=30, ax=ax)
    else:
        X[feat].value_counts(dropna=False).head(25).plot(kind="bar", ax=ax)
    ax.set_xlabel(feat)
    ax.set_ylabel("Count")
    st.pyplot(fig)

with c2:
    st.subheader("Box Plot")
    fig, ax = plt.subplots()
    if pd.api.types.is_numeric_dtype(X[feat]):
        sns.boxplot(x=X[feat].dropna(), ax=ax)
        ax.set_xlabel(feat)
        ax.set_ylabel("")
    else:
        st.info("Box plot not applicable for categorical feature.")
    st.pyplot(fig)

# -----------------------------
# Train/Validation
# -----------------------------
st.markdown("### Train Baseline Model")

# Stratify only for small-number classes (binary/low-cardinality targets)
stratify_arg = y if y.nunique() <= 20 else None

test_size = st.slider("Test size", 0.10, 0.40, 0.20, 0.05)
random_state = 42

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state, stratify=stratify_arg
)

# Preprocess & model
pre = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ],
    remainder="drop",
)

clf = LogisticRegression(max_iter=1000)

pipe = Pipeline(steps=[
    ("preprocess", pre),
    ("clf", clf),
])

pipe.fit(X_train, y_train)

# Inference
y_pred = pipe.predict(X_test)
y_prob = None
try:
    y_prob = pipe.predict_proba(X_test)[:, 1]
except Exception:
    pass

# -----------------------------
# Metrics
# -----------------------------
st.markdown("#### Metrics")

report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
rep_df = pd.DataFrame(report).T.round(3)
st.dataframe(rep_df, use_container_width=True)

if (y_prob is not None) and (y_test.nunique() == 2):
    try:
        auc = roc_auc_score(y_test, y_prob)
        st.write(f"**ROC AUC:** {auc:.3f}")
    except Exception:
        pass

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cbar=False, ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix")
st.pyplot(fig)

# -----------------------------
# Score new CSV
# -----------------------------
st.markdown("---")
st.subheader("Score New Data")
new_file = st.file_uploader("Upload a CSV with the **same feature schema** (no target column)", type=["csv"], key="score_uploader")

if new_file:
    new_df = _read_csv_any(new_file)
    new_df.columns = new_df.columns.str.strip().str.replace(r"\s+", "_", regex=True)

    # align incoming columns to training schema:
    # keep columns present in training features; ignore extras
    missing_cols = [c for c in X.columns if c not in new_df.columns]
    for mc in missing_cols:
        new_df[mc] = np.nan
    new_df = new_df[X.columns]  # reorder

    try:
        scored = new_df.copy()
        preds = pipe.predict(new_df)
        scored["prediction"] = preds
        if hasattr(pipe.named_steps["clf"], "predict_proba"):
            scored["score"] = pipe.predict_proba(new_df)[:, 1]
        st.dataframe(scored.head(), use_container_width=True)
        st.download_button(
            "Download scored file",
            data=scored.to_csv(index=False),
            file_name="scored_car_insurance.csv",
            mime="text/csv",
        )
    except Exception as e:
        st.error(f"Could not score the file: {e}")

# -----------------------------
# Notes
# -----------------------------
with st.expander("Notes", expanded=False):
    st.markdown(
        """
- This is a **baseline** Logistic Regression. You can swap in other estimators in the pipeline.
- For production, pair model scores with **rule-based checks** (e.g., documentation flags).
- If your CSV uses `;` as a delimiter, the app auto-tries alternate parsing.
"""
    )
