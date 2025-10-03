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
st.set_page_config(page_title="Best Single Feature", page_icon="📈", layout="wide")
st.title("🚗 Car Insurance Risk Factors — Logistic Regression Feature Benchmark")

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
st.markdown("Insurance companies invest a lot of time and money into optimizing their pricing and accurately estimating the likelihood that customers will make a claim. In many countries insurance it is a legal requirement to have car insurance in order to drive a vehicle on public roads, so the market is very large!. Knowing all of this, On the Road car insurance have requested your services in building a model to predict whether a customer will make a claim on their insurance during the policy period. As they have very little expertise and infrastructure for deploying and monitoring machine learning models, they've asked you to identify the single feature that results in the best performing model, as measured by accuracy, so they can start with a simple model in production.")

st.header("Objective")
st.markdown("Identify the single feature of the data that is the best predictor of whether a customer will put in a claim.")

st.header("About the Data")
st.markdown("They have supplied their customer data as a csv file called car_insurance.csv, along with a table detailing the column names and descriptions below.")
c0, c1 = st.columns([2, 1])

with c0:
    st.subheader("Preview")
    st.dataframe(df.head(), use_container_width=True)
with c1:
    st.subheader("Shape & Missingness")
    st.metric("Rows", len(df))
    st.metric("Columns", df.shape[1])
    st.metric("Overall Missing %", f"{(df.isna().mean().mean()*100):.2f}%")

# Concise schema table
schema_rows = [
    ("id", "Unique client ID"),
    ("age", "Client's age (0:16–25, 1:26–39, 2:40–64, 3:65+)"),
    ("gender", "Client's gender (0:Female, 1:Male)"),
    ("driving_experience", "Years driving (0:0–9, 1:10–19, 2:20–29, 3:30+)"),
    ("education", "Education (0:None, 1:High school, 2:University)"),
    ("income", "Income (0:Poverty, 1:Working, 2:Middle, 3:Upper)"),
    ("credit_score", "Credit score (0–1)"),
    ("vehicle_ownership", "Owns vehicle? (0:Financing, 1:Owns)"),
    ("vehicle_year", "Registration year (0:<2015, 1:≥2015)"),
    ("married", "Marital status (0:Not married, 1:Married)"),
    ("children", "Number of children"),
    ("postal_code", "Postal code"),
    ("annual_mileage", "Miles driven per year"),
    ("vehicle_type", "Type (0:Sedan, 1:Sports)"),
    ("speeding_violations", "Count of speeding violations"),
    ("duis", "Count of DUIs"),
    ("past_accidents", "Count of past accidents"),
    ("outcome", "Insurance claim (0:No, 1:Yes)"),
]

schema_df = pd.DataFrame(schema_rows, columns=["Column", "Description"])

st.subheader("Dataset Dictionary (Concise)")
st.table(schema_df)


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
