# app.py — Best Single Feature by Accuracy using statsmodels Logit (matches Colab loop)
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns


import statsmodels.api as sm
from statsmodels.formula.api import logit

# ---------------------------------------------------------
# Page setup
# ---------------------------------------------------------
st.set_page_config(page_title="Best Single Feature (Logit)", page_icon="📈", layout="wide")
st.title("📈 Best Single Feature — Logistic Regression (Full Data)")
st.caption("Replicates the Colab loop: outcome ~ feature → pred_table() → Accuracy")

# ---------------------------------------------------------
# Data loading (fixed file with uploader fallback)
# ---------------------------------------------------------
DATA_PATH = Path("data/car_insurance.csv")   # adjust if needed
DEP_VAR = "outcome"                           # <- Colab’s dependent variable name

@st.cache_data(show_spinner=False)
def read_csv_forgiving(src):
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
        up = st.file_uploader("Upload your dataset (CSV)", type=["csv"])
        if not up:
            st.info("Add a dataset at data/car_insurance.csv or upload a CSV.")
            st.stop()
        df = read_csv_forgiving(up)
        src = "uploaded file"
    # Keep original column names for formula; also create a normalized alias for display
    return df, src

df, src = load_data()
st.success(f"Loaded dataset from {src}")

# ---------------------------------------------------------
# Context / Objective / About the Data (brief)
# ---------------------------------------------------------
st.header("Context")
st.markdown("Identify which **single variable** best predicts the binary **outcome** using a simple **Logit** model per feature.")

st.header("Objective")
st.markdown("For each candidate feature `X`, fit `logit('outcome ~ X')` on **full data** and compute **Accuracy** from `pred_table()`.")

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

# ---------------------------------------------------------
# Guard rails
# ---------------------------------------------------------
if DEP_VAR not in df.columns:
    st.error(f"Dependent variable `{DEP_VAR}` not found. Rename your target column to `{DEP_VAR}` (as in Colab) or change DEP_VAR in app.py.")
    st.stop()

# Ensure outcome is binary-like (0/1). If it’s 'Yes/No' etc, try to map automatically.
y_raw = df[DEP_VAR]
y_mapped = y_raw.copy()

if y_raw.dropna().dtype == object:
    # attempt mapping of common labels
    lower = y_raw.str.lower()
    if set(lower.dropna().unique()) <= {"yes", "no"}:
        y_mapped = lower.map({"no": 0, "yes": 1})
    elif set(lower.dropna().unique()) <= {"true", "false"}:
        y_mapped = lower.map({"false": 0, "true": 1})
    else:
        # leave as-is; statsmodels can handle if already 0/1 strings
        pass

df = df.copy()
df[DEP_VAR] = y_mapped

# ---------------------------------------------------------
# Actions — Data Preparation
# ---------------------------------------------------------
st.header("Actions — Data Preparation")
st.markdown("""
- Keep column names as-is for the statsmodels formula interface.  
- Drop exact duplicate rows.  
- For each feature `col` (excluding `outcome`):  
  1) Try `logit("outcome ~ col", data=df).fit()`  
  2) If it fails (categorical), retry with `logit("outcome ~ C(col)", data=df).fit()`  
  3) Use `model.pred_table()` to extract **Accuracy** at threshold 0.5.  
""")

df_clean = df.drop_duplicates().copy()
before, after = len(df), len(df_clean)
st.caption(f"Removed {before - after} duplicate rows (kept {after}).")

# ---------------------------------------------------------
# Results — loop exactly like Colab
# ---------------------------------------------------------
st.header("Results — Best Single Feature (Accuracy)")

# Candidate features: all columns except the dependent var
features = [c for c in df_clean.columns if c != DEP_VAR]

models = []
accuracies = []
used_features = []

for col in features:
    # Drop rows with NA in outcome or this feature
    sub = df_clean[[DEP_VAR, col]].dropna().copy()
    if sub.empty or sub[DEP_VAR].dropna().nunique() < 2:
        # cannot fit a logit if outcome has <2 classes after dropping NA
        continue

    # Try numeric/formula directly first
    formula = f"{DEP_VAR} ~ {col}"
    model = None
    try:
        model = logit(formula, data=sub).fit(disp=False)
    except Exception:
        # Retry as categorical feature
        try:
            formula = f"{DEP_VAR} ~ C({col})"
            model = logit(formula, data=sub).fit(disp=False)
        except Exception:
            model = None

    if model is None:
        continue

    # pred_table() returns [[tn, fp],[fn, tp]] at default threshold 0.5
    try:
        conf_matrix = model.pred_table()
        tn = conf_matrix[0, 0]
        fp = conf_matrix[0, 1]
        fn = conf_matrix[1, 0]
        tp = conf_matrix[1, 1]
        acc = (tn + tp) / (tn + fp + fn + tp)
    except Exception:
        acc = np.nan

    models.append(model)
    accuracies.append(acc)
    used_features.append(col)

# Build results frame
if not accuracies:
    st.error("Could not fit a valid Logit for any single feature. Check that `outcome` is binary-like and features have variation.")
    st.stop()

best_idx = int(np.nanargmax(accuracies))
best_feature = used_features[best_idx]
best_accuracy = float(accuracies[best_idx])

best_feature_df = pd.DataFrame(
    {"best_feature": [best_feature], "best_accuracy": [best_accuracy]}
)

st.subheader("Best Feature")
st.dataframe(best_feature_df.round(3), use_container_width=True)

# Full ranking
rank_df = pd.DataFrame({"feature": used_features, "accuracy": accuracies})
rank_df = rank_df.sort_values("accuracy", ascending=False).reset_index(drop=True)

st.subheader("All Features — Ranked by Accuracy (Logit, full data)")
st.dataframe(rank_df.round(3), use_container_width=True)

# Optional small bar chart (top 10)
top_n = min(10, len(rank_df))
fig, ax = plt.subplots(figsize=(7, max(3, int(top_n*0.4))))
plot_df = rank_df.head(top_n).sort_values("accuracy")
ax.barh(plot_df["feature"], plot_df["accuracy"])
ax.set_xlabel("Accuracy (pred_table threshold 0.5)")
ax.set_ylabel("")
st.pyplot(fig)

st.markdown("---")
st.caption("This app mirrors the Colab loop: per-feature Logit on full data, accuracy via pred_table().")
