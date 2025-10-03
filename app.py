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
with st.expander("Context & Problem Statement", expanded=True):
    st.markdown("""
    **Context**  
    Insurance companies invest a lot of time and money into optimizing their pricing and accurately estimating the likelihood that customers will make a claim. In many countries insurance it is a legal requirement to have car insurance in order to drive a vehicle on public roads, so the market is very large!. Knowing all of this, On the Road car insurance have requested your services in building a model to predict whether a customer will make a claim on their insurance during the policy period. As they have very little expertise and infrastructure for deploying and monitoring machine learning models, they've asked you to identify the single feature that results in the best performing model, as measured by accuracy, so they can start with a simple model in production.
    
    **Problem Statement**  
    Identify the single feature of the data that is the best predictor of whether a customer will put in a claim.

    **About the Data**
    They have supplied their customer data as a csv file called car_insurance.csv, along with a table detailing the column names and descriptions below.
    """)

schema_rows = [
    ("id", "client ID"),
    ("age", "0=16–25, 1=26–39, 2=40–64, 3=65+"),
    ("gender", "0=F, 1=M"),
    ("driving_experience", "0=0–9, 1=10–19, 2=20–29, 3=30+"),
    ("education", "0=None, 1=HS, 2=Univ"),
    ("income", "0=Poverty, 1=Working, 2=Middle, 3=Upper"),
    ("credit_score", "0–1"),
    ("vehicle_ownership", "0=Financing, 1=Owns"),
    ("vehicle_year", "0=<2015, 1=≥2015"),
    ("married", "0=No, 1=Yes"),
    ("children", "count"),
    ("postal_code", "code"),
    ("annual_mileage", "miles/year"),
    ("vehicle_type", "0=Sedan, 1=Sports"),
    ("speeding_violations", "count"),
    ("duis", "count"),
    ("past_accidents", "count"),
    ("outcome", "0=No claim, 1=Claim"),
]
schema_df = pd.DataFrame(schema_rows, columns=["Column", "Meaning"])

with st.expander("📑 Dataset Details:", expanded=True):
    st.table(schema_df)

    
st.subheader("Raw Preview")
st.dataframe(df.head(), use_container_width=True)

st.subheader("Shape & Missingness")
def metric_card(title, value, subtitle=None, bg="#0ea5e9", fg="white"):
    """
    Renders a single metric 'card' with custom background color.

    title: str (small heading)
    value: str/number (big text)
    subtitle: str (tiny helper text)
    bg: background color (hex or CSS color)
    fg: foreground/text color
    """
    st.markdown(
        f"""
        <div style="
            border-radius:16px;
            padding:16px 18px;
            background:{bg};
            color:{fg};
            box-shadow: 0 6px 20px rgba(0,0,0,0.08);
            border: 1px solid rgba(255,255,255,0.15);
            ">
            <div style="font-size:13px; opacity:0.9; letter-spacing:.4px; text-transform:uppercase;">
                {title}
            </div>
            <div style="font-size:30px; font-weight:700; margin-top:6px;">
                {value}
            </div>
            <div style="font-size:12px; opacity:0.85; margin-top:4px;">
                {subtitle or ""}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# Example usage — three cards in a row
rows = f"{len(df):,}"
cols = f"{df.shape[1]:,}"
miss = f"{(df.isna().mean().mean()*100):.2f}%"

c1, c2, c3 = st.columns(3)
with c1:
    metric_card("Rows", rows, "Total records", bg="#1f6feb")         # Blue
with c2:
    metric_card("Columns", cols, "Total fields", bg="#14b8a6")       # Green
with c3:
    metric_card("Missing %", miss, "Overall null share", bg="#f59e0b") # Amber

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
- Removed duplicate rows.  
- Excluded non-informative identifiers from the feature set: `id`.  
- Excluded the dependent variable from the feature set: `outcome`.  
- Imputed missing values for numeric fields using column means:  
  - `credit_score` ← mean(`credit_score`)  
  - `annual_mileage` ← mean(`annual_mileage`)  
""")

# Clean duplicates
df_clean = df.drop_duplicates().copy()
before, after = len(df), len(df_clean)
st.caption(f"Removed {before - after} duplicate rows (kept {after}).")

# Impute missing values
for col in ["credit_score", "annual_mileage"]:
    if col in df_clean.columns:
        df_clean[col].fillna(df_clean[col].mean(), inplace=True)

# Drop non-informative columns from feature set
ID_COLS = ["id"]
drop_from_features = [DEP_VAR] + [c for c in ID_COLS if c in df_clean.columns]
df_model = df_clean.drop(columns=drop_from_features, errors="ignore")

st.caption(
    "Dropped from feature set: "
    + (", ".join([c for c in drop_from_features if c in df_clean.columns]) or "(none)")
)

# ---------------------------------------------------------
# Results — Best Single Feature basis Accuracy 
# ---------------------------------------------------------
st.header("Results — Best Single Feature basis Accuracy")

features = list(df_model.columns)   # Only model features left
models, accuracies, used_features = [], [], []

for col in features:
    sub = pd.concat([df_clean[[DEP_VAR]], df_model[[col]]], axis=1).dropna().copy()
    if sub.empty or sub[DEP_VAR].nunique() < 2:
        continue

    model = None
    try:
        model = logit(f"{DEP_VAR} ~ {col}", data=sub).fit(disp=False)
    except Exception:
        try:
            model = logit(f"{DEP_VAR} ~ C({col})", data=sub).fit(disp=False)
        except Exception:
            model = None
    if model is None:
        continue

    try:
        cm = model.pred_table()
        tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
        acc = (tn + tp) / (tn + fp + fn + tp)
    except Exception:
        acc = np.nan

    models.append(model)
    accuracies.append(acc)
    used_features.append(col)

if not accuracies:
    st.error("Could not fit a valid Logit for any feature.")
    st.stop()

# Best feature
best_idx = int(np.nanargmax(accuracies))
best_feature = used_features[best_idx]
best_accuracy = float(accuracies[best_idx])

def highlight_card(title, value, sub=None, bg="#22c55e", fg="white"):
    st.markdown(
        f"""
        <div style="
            border-radius:16px;
            padding:16px 18px;
            background:{bg};
            color:{fg};
            box-shadow: 0 6px 20px rgba(0,0,0,0.10);
            border: 1px solid rgba(255,255,255,0.12);
            ">
            <div style="font-size:13px; opacity:0.9; letter-spacing:.4px; text-transform:uppercase;">
                {title}
            </div>
            <div style="font-size:30px; font-weight:700; margin-top:6px;">
                {value}
            </div>
            <div style="font-size:24px; margin-top:4px;">
                {sub or ""}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

highlight_card("🏆 Best Feature", f"{best_feature}", f"Accuracy: {best_accuracy:.3f}", bg="#22c55e")

# ---------- Full ranking (lighter colors) ----------
# =========================
# Rank variables by Accuracy (statsmodels Logit) + Bar Chart
# Requirements in scope: df_clean (DataFrame), DEP_VAR (str: dependent var name)
# =========================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.formula.api import logit
import streamlit as st

# 1) Build the feature set (drop outcome + obvious identifiers)
ID_COLS = ["id"]
feature_df = df_clean.drop(columns=[c for c in [DEP_VAR, *ID_COLS] if c in df_clean.columns], errors="ignore")

# 2) Compute per-feature accuracy via logit("outcome ~ feature")
used_features, accuracies = [], []
for col in feature_df.columns:
    sub = pd.concat([df_clean[[DEP_VAR]], feature_df[[col]]], axis=1).dropna()
    if sub.empty or sub[DEP_VAR].nunique() < 2:
        continue

    model = None
    # try as numeric
    try:
        model = logit(f"{DEP_VAR} ~ {col}", data=sub).fit(disp=False)
    except Exception:
        # fallback: treat as categorical
        try:
            model = logit(f"{DEP_VAR} ~ C({col})", data=sub).fit(disp=False)
        except Exception:
            model = None
    if model is None:
        continue

    try:
        cm = model.pred_table()  # [[tn, fp], [fn, tp]] at 0.5
        tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
        acc = (tn + tp) / (tn + fp + fn + tp)
    except Exception:
        acc = np.nan

    used_features.append(col)
    accuracies.append(acc)

if not accuracies:
    st.error("No valid single-feature Logit models could be fit.")
    st.stop()

# 3) Build and show the ranking table (light, pretty)
rank_df = pd.DataFrame({"Feature": used_features, "Accuracy": accuracies})
rank_df = rank_df.sort_values("Accuracy", ascending=False).reset_index(drop=True)
best_feature = rank_df.iloc[0]["Feature"]

st.subheader("All Features — Ranked by Accuracy")

def zebra_light(data, even="#f9fafb", odd="#ffffff", font="#111827"):
    styles = pd.DataFrame("", index=data.index, columns=data.columns)
    even_mask = (np.arange(len(data)) % 2 == 0)
    styles.loc[even_mask, :] = f"background-color: {even}; color: {font};"
    styles.loc[~even_mask, :] = f"background-color: {odd}; color: {font};"
    return styles

def highlight_best_row(data, best_value, color="#d1fae5", font="#065f46"):
    styles = pd.DataFrame("", index=data.index, columns=data.columns)
    if "Feature" in data.columns:
        mask = data["Feature"] == best_value
        styles.loc[mask, :] = f"background-color: {color}; color: {font}; font-weight:600;"
    return styles

table_disp = rank_df.copy()
table_disp[""] = np.where(table_disp["Feature"] == best_feature, "⭐", "")
table_disp = table_disp[["", "Feature", "Accuracy"]].round({"Accuracy": 3})

try:
    rank_sty = (
        table_disp.style
            .set_table_styles([
                {"selector": "th", "props": [
                    ("text-align", "left"),
                    ("background", "#eef2ff"),
                    ("color", "#111827"),
                    ("padding", "8px 10px"),
                    ("border", "0px")
                ]},
                {"selector": "td", "props": [
                    ("padding", "8px 10px"),
                    ("border", "0px")
                ]}
            ])
            .hide(axis="index")
            .format({"Accuracy": "{:.3f}"})
            .bar(subset=["Accuracy"], color="#93c5fd", vmin=0, vmax=1)
            .apply(zebra_light, axis=None)
            .apply(lambda d: highlight_best_row(d, best_feature), axis=None)
    )
    st.dataframe(rank_sty, use_container_width=True)
except Exception:
    st.dataframe(table_disp, use_container_width=True)

st.download_button(
    "⬇️ Download feature accuracies (CSV)",
    data=rank_df.to_csv(index=False),
    file_name="feature_accuracies.csv",
    mime="text/csv",
)

# 4) Bar chart (Top N)
st.subheader("Accuracy — Top Features")
max_n = min(20, len(rank_df))
top_n = st.slider("Show top N features", min_value=5, max_value=max_n, value=min(10, max_n), step=1)

plot_df = rank_df.head(top_n).sort_values("Accuracy")
fig, ax = plt.subplots(figsize=(8, max(3.5, 0.45*top_n)))
bars = ax.barh(plot_df["Feature"], plot_df["Accuracy"], color="#93c5fd")

ax.set_xlabel("Accuracy (pred_table threshold 0.5)")
ax.set_xlim(0, 1)
ax.grid(axis="x", linestyle=":", alpha=0.35)

for i, (feat, acc) in enumerate(zip(plot_df["Feature"], plot_df["Accuracy"])):
    ax.text(acc + 0.01, i, f"{acc:.3f}", va="center", color="#374151")

if best_feature in plot_df["Feature"].values:
    idx = plot_df.index[plot_df["Feature"] == best_feature][0]
    bars[list(plot_df.index).index(idx)].set_edgecolor("#2563eb")
    bars[list(plot_df.index).index(idx)].set_linewidth(2)

st.pyplot(fig)
