import streamlit as st
import pandas as pd
import numpy as np

from pathlib import Path
from io import StringIO

# Charts / model
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="Car Insurance — Claims Classifier", page_icon="🚗", layout="wide")

# Banner
banner = Path("assets/car.jpg")
if banner.exists():
    st.image(str(banner), use_container_width=True)

st.title("🚗 Car Insurance — Claims Risk Classifier")
st.caption("Context → Objective → Data → Actions → Results")

# -----------------------------
# Context & Objective
# -----------------------------
with st.expander("Context & Objective", expanded=True):
    st.markdown("""
**Context.** Insurers must triage claims and spot risky patterns early (fraud, high loss likelihood), while keeping straight-through processing fast for good customers.

**Objective.** Build a lightweight, reproducible **classification app** that:
- explores the car-insurance dataset,
- trains a baseline model (Logistic Regression) to predict a **target** (e.g., claim outcome/fraud flag),
- lets you **score new CSVs** and download results.
""")

# -----------------------------
# Load data (fixed file)
# -----------------------------
DATA_PATH = Path("data/car_insurance.csv")
if not DATA_PATH.exists():
    st.error("`data/car_insurance.csv` not found. Add the CSV to `data/` and re-run.")
    st.stop()

@st.cache_data(show_spinner=False)
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # normalize column names a bit
    df.columns = (
        df.columns
          .str.strip()
          .str.replace(r"\s+", "_", regex=True)
    )
    return df

df = load_data(DATA_PATH)
st.subheader("Data Preview")
st.dataframe(df.head(), use_container_width=True)

# -----------------------------
# Select target + feature audit
# -----------------------------
# Heuristic: try to guess a binary target if present, else ask user
DEFAULT_TARGETS = [c for c in df.columns if c.lower() in ("fraud_found", "is_fraud", "fraud_flag", "claim", "claim_status", "target")]
target = st.selectbox("Select target column (binary 0/1 preferred)", options=df.columns, index=0 if DEFAULT_TARGETS==[] else df.columns.get_loc(DEFAULT_TARGETS[0]))

if target not in df.columns:
    st.error("Select a valid target column.")
    st.stop()

# Separate features/target
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
# EDA (quick)
# -----------------------------
st.markdown("### Exploratory Analysis")
feat = st.selectbox("Select a feature to explore", options=X.columns)

c1, c2 = st.columns(2)
with c1:
    st.subheader("Histogram")
    fig, ax = plt.subplots()
    if pd.api.types.is_numeric_dtype(X[feat]):
        sns.histplot(X[feat].dropna(), bins=30, ax=ax)
    else:
        X[feat].value_counts().plot(kind="bar", ax=ax)
    ax.set_xlabel(feat); ax.set_ylabel("Count")
    st.pyplot(fig)

with c2:
    st.subheader("Box Plot")
    fig, ax = plt.subplots()
    if pd.api.types.is_numeric_dtype(X[feat]):
        sns.boxplot(x=X[feat].dropna(), ax=ax)
        ax.set_xlabel(feat); ax.set_ylabel("")
    else:
        st.info("Box plot not applicable for categorical feature.")
    st.pyplot(fig)

# -----------------------------
# Train/Test + Pipeline
# -----------------------------
st.markdown("### Train Baseline Model")

test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
random_state = 42

# Preprocess
pre = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ],
    remainder="drop"
)

clf = LogisticRegression(max_iter=1000, n_jobs=None)

pipe = Pipeline(steps=[
    ("preprocess", pre),
    ("clf", clf)
])

# Train/validate
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y if y.nunique() <= 20 else None)

pipe.fit(X_train, y_train)
y_prob = None
try:
    y_prob = pipe.predict_proba(X_test)[:, 1]
except Exception:
    pass
y_pred = pipe.predict(X_test)

# Metrics
st.markdown("#### Metrics")
report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
rep_df = pd.DataFrame(report).T.round(3)
st.dataframe(rep_df, use_container_width=True)

if y_prob is not None and y_test.nunique() == 2:
    auc = roc_auc_score(y_test, y_prob)
    st.write(f"**ROC AUC:** {auc:.3f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cbar=False, ax=ax)
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion Matrix")
st.pyplot(fig)

# -----------------------------
# Score new CSV
# -----------------------------
st.markdown("---")
st.subheader("Score New Data (optional)")
uploaded = st.file_uploader("Upload a CSV with the same schema (features only)", type=["csv"])
if uploaded:
    new_df = pd.read_csv(uploaded)
    new_df.columns = (
        new_df.columns
          .str.strip()
          .str.replace(r"\s+", "_", regex=True)
    )
    try:
        preds = pipe.predict(new_df)
        out = new_df.copy()
        out["prediction"] = preds
        if hasattr(pipe.named_steps["clf"], "predict_proba"):
            out["score"] = pipe.predict_proba(new_df)[:, 1]
        st.dataframe(out.head(), use_container_width=True)
        st.download_button("Download scored file", out.to_csv(index=False), file_name="scored_car_insurance.csv")
    except Exception as e:
        st.error(f"Could not score the file: {e}")

# -----------------------------
# Notes
# -----------------------------
with st.expander("Notes", expanded=False):
    st.markdown("""
- This is a **baseline** model for demonstration. Swap `LogisticRegression` for `RandomForestClassifier` / `XGBClassifier` if needed.
- Put domain rules on top of model scores for production (e.g., hard fraud triggers, documentation checks).
""")
