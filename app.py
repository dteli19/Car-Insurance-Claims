# app.py — Best Single Feature by Accuracy (Logistic Regression, full data, no target detection)
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# -----------------------------
# Settings
# -----------------------------
st.set_page_config(page_title="Car Insurance — Best Feature", page_icon="🚗", layout="wide")
st.title("🚗 Car Insurance — Best Single Feature (Accuracy)")
st.caption("Context • Objective • Data • Actions • Observations • Results")

DATA_PATH = Path("data/car_insurance.csv")
TARGET_COL = "Response"  # <-- set this to your Colab target column (no auto-detection)

# -----------------------------
# Load data
# -----------------------------
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
        up = st.file_uploader("Upload a CSV", type=["csv"])
        if not up:
            st.info("Add `data/car_insurance.csv` or upload your dataset to proceed.")
            st.stop()
        df = read_csv_forgiving(up)
        src = "uploaded file"
    df.columns = df.columns.str.strip().str.replace(r"\s+", "_", regex=True)
    return df, src

df, src = load_data()
st.success(f"Loaded dataset from {src}")

# -----------------------------
# Context & Objective
# -----------------------------
st.header("Context")
st.markdown("Identify which **single feature** best predicts the target in a car insurance dataset.")

st.header("Objective")
st.markdown("Fit **Logistic Regression on full data** (no train/test) per-feature and choose the **best by Accuracy**.")

# -----------------------------
# About the Data
# -----------------------------
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

num_cols_all = df.select_dtypes(include=[np.number]).columns.tolist()
st.subheader("Summary (Numeric Features)")
if num_cols_all:
    desc = df[num_cols_all].agg(["count","mean","median","std","min","max"]).T
    q = df[num_cols_all].quantile([0.25,0.75])
    desc["q1"] = q.loc[0.25].values
    desc["q3"] = q.loc[0.75].values
    desc["iqr"] = desc["q3"] - desc["q1"]
    desc["missing_%"] = (df[num_cols_all].isna().mean()*100).values
    st.dataframe(desc[["count","mean","median","std","min","q1","q3","iqr","max","missing_%"]].round(3),
                 use_container_width=True)
else:
    st.info("No numeric columns found.")

# -----------------------------
# Actions — Data Preparation
# -----------------------------
st.header("Actions — Data Preparation")
st.markdown("""
1. Normalize headers (trim, underscores).  
2. Drop exact duplicate rows.  
3. For each single feature:  
   - Numeric → `StandardScaler`  
   - Categorical → `OneHotEncoder(handle_unknown="ignore")`  
   - Model → `LogisticRegression(max_iter=1000)` on **full data** (resubstitution).
""")
before = len(df)
df_clean = df.drop_duplicates().copy()
st.caption(f"Removed {before - len(df_clean)} duplicate rows.")

# -----------------------------
# Observations (light)
# -----------------------------
st.header("Observations")
obs = []
if num_cols_all:
    skew = df_clean[num_cols_all].skew(numeric_only=True).sort_values(ascending=False)
    skewed = [f"{c} (skew={v:.2f})" for c, v in skew.head(3).items() if v > 1.0]
    if skewed: obs.append("Right-skew detected: " + ", ".join(skewed) + ".")
for col in num_cols_all[:5]:
    s = df_clean[col].dropna()
    if len(s) > 0:
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        obs.append(f"**{col}** typically spans **{q1:,.2f}–{q3:,.2f}** (IQR).")
for o in (obs or ["- No notable skew/range signals detected."]):
    st.markdown(f"- {o}")

# -----------------------------
# Results — Best Single Feature by Accuracy
# -----------------------------
st.header("Results — Best Single Feature (Accuracy)")

if TARGET_COL not in df_clean.columns:
    st.error(f"Target column `{TARGET_COL}` not found. Set `TARGET_COL` near the top of app.py to your Colab target.")
    st.stop()

y = df_clean[TARGET_COL].dropna()
X = df_clean.loc[y.index].drop(columns=[TARGET_COL])

# drop constant/empty columns
def is_constant(s: pd.Series) -> bool:
    s_non_null = s.dropna()
    return s_non_null.nunique() <= 1

features = [c for c in X.columns if not is_constant(X[c])]

if not features:
    st.error("No valid feature columns available after cleaning.")
    st.stop()

num_cols = X[features].select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in features if c not in num_cols]

def eval_feature(feat: str):
    Xi = X[[feat]]
    if feat in num_cols:
        pre = ColumnTransformer([("num", StandardScaler(), [feat])], remainder="drop")
    else:
        pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), [feat])], remainder="drop")
    pipe = Pipeline([("preprocess", pre), ("clf", LogisticRegression(max_iter=1000))])
    pipe.fit(Xi, y)
    y_pred = pipe.predict(Xi)
    return accuracy_score(y, y_pred)

results = []
for f in features:
    try:
        acc = eval_feature(f)
        results.append({"feature": f, "type": "numeric" if f in num_cols else "categorical", "accuracy": acc})
    except Exception:
        results.append({"feature": f, "type": "error", "accuracy": np.nan})

res_df = pd.DataFrame(results).sort_values("accuracy", ascending=False).reset_index(drop=True)

if res_df.empty or res_df["accuracy"].isna().all():
    st.error("Unable to compute accuracy for any feature. Check target and feature columns.")
    st.stop()

best = res_df.iloc[0]
st.subheader("Best Feature")
st.markdown(
    f"- **Feature:** `{best['feature']}`  \n"
    f"- **Type:** {best['type']}  \n"
    f"- **Accuracy:** **{best['accuracy']:.3f}**"
)

# Ranked table
st.subheader("All Features — Ranked by Accuracy")
disp = res_df.copy().round({"accuracy": 3})
styler = disp.style.bar(subset=["accuracy"], color="#76b7ff").hide(axis="index")
st.dataframe(styler, use_container_width=True)

# Optional: small bar chart of top N
top_n = min(10, len(disp))
fig, ax = plt.subplots(figsize=(7, max(3, int(top_n*0.4))))
plot_df = disp.head(top_n).sort_values("accuracy")
ax.barh(plot_df["feature"], plot_df["accuracy"])
ax.set_xlabel("Accuracy")
ax.set_ylabel("")
st.pyplot(fig)

st.markdown("---")
st.caption("Full-data (resubstitution) Logistic Regression. Adjust TARGET_COL to match your Colab target.")
