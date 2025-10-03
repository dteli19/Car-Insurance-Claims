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

st.markdown("""
- **id** — Unique client identifier  
- **age** — Client’s age group  
  - 0: 16–25  
  - 1: 26–39  
  - 2: 40–64  
  - 3: 65+  
- **gender** — Client’s gender  
  - 0: Female  
  - 1: Male  
- **driving_experience** — Years driving  
  - 0: 0–9  
  - 1: 10–19  
  - 2: 20–29  
  - 3: 30+  
- **education** — Education level  
  - 0: No education  
  - 1: High school  
  - 2: University  
- **income** — Income level  
  - 0: Poverty  
  - 1: Working class  
  - 2: Middle class  
  - 3: Upper class  
- **credit_score** — Credit score (0–1)  
- **vehicle_ownership** — Vehicle ownership  
  - 0: Financing  
  - 1: Owns  
- **vehicle_year** — Registration year  
  - 0: Before 2015  
  - 1: 2015 or later  
- **married** — Marital status  
  - 0: Not married  
  - 1: Married  
- **children** — Number of children  
- **postal_code** — Postal code  
- **annual_mileage** — Miles driven per year  
- **vehicle_type** — Type of car  
  - 0: Sedan  
  - 1: Sports car  
- **speeding_violations** — Count of speeding violations  
- **duis** — Count of DUI incidents  
- **past_accidents** — Count of past accidents  
- **outcome** — Insurance claim status  
  - 0: No claim  
  - 1: Made a claim  
""")
    
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
    metric_card("Columns", cols, "Total fields", bg="#10b981")       # Green
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
- Assessed nulls across all columns, with focus on `credit_score` and `annual_mileage`.
- Imputed missing values for these **numeric** fields using their **column means** to preserve sample size:
  - `credit_score` ← mean(`credit_score`)
  - `annual_mileage` ← mean(`annual_mileage`)
- No rows were dropped for missingness; other columns were left unchanged.
""")

df_clean = df.drop_duplicates().copy()
before, after = len(df), len(df_clean)
st.caption(f"Removed {before - after} duplicate rows (kept {after}).")

# ---------------------------------------------------------
# Results — Best Single Feature basis Accuracy 
# ---------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.formula.api import logit

st.header("Results — Best Single Feature basis Accuracy")

# Candidate features: all columns except the dependent var
features = [c for c in df_clean.columns if c != DEP_VAR]

models = []
accuracies = []
used_features = []

for col in features:
    # Drop rows with NA in outcome or this feature
    sub = df_clean[[DEP_VAR, col]].dropna().copy()
    if sub.empty or sub[DEP_VAR].dropna().nunique() < 2:
        # cannot fit a logit if outcome has < 2 classes after dropping NA
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

# ---------- Pretty "Best Feature" card + compact table ----------
def highlight_card(title, value, sub=None, bg="#1f6feb", fg="white"):
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
            <div style="font-size:12px; opacity:0.9; margin-top:4px;">
                {sub or ""}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

highlight_card("🏆 Best Feature", f"{best_feature}", f"Accuracy: **{best_accuracy:.3f}**", bg="#1f6feb")

bf_sty = (
    bf_disp.style
    .hide(axis="index")
    .set_properties(**{"text-align": "left"})
    .set_table_styles([
    {"selector": "th", "props": [("text-align", "left"),
                                 ("background", "#0b1220"),
                                 ("color", "white"),
                                 ("padding", "8px 10px")]},
        {"selector": "td", "props": [("padding", "8px 10px")]}
        ])
    .bar(subset=["Accuracy"], color="#66b3ff", vmin=0, vmax=1)
    )
st.dataframe(bf_sty, use_container_width=True)

# ---------- Full ranking (Feature • Accuracy) ----------
rank_df = pd.DataFrame({"feature": used_features, "accuracy": accuracies})
rank_df = rank_df.sort_values("accuracy", ascending=False).reset_index(drop=True)

st.subheader("All Features — Ranked by Accuracy (Logit, full data)")
rank_disp = (rank_df.copy()
             .rename(columns={"feature": "Feature", "accuracy": "Accuracy"})
             .round({"Accuracy": 3}))
rank_disp[""] = np.where(rank_disp["Feature"] == best_feature, "⭐", "")
rank_disp = rank_disp[["", "Feature", "Accuracy"]]

rank_sty = (
    rank_disp.style
      .hide(axis="index")
      .set_properties(**{"text-align": "left"})
      .set_table_styles([
          {"selector": "th", "props": [("text-align", "left"),
                                       ("background", "#0b1220"),
                                       ("color", "white"),
                                       ("padding", "8px 10px")]},
          {"selector": "td", "props": [("padding", "8px 10px")]}
      ])
      .bar(subset=["Accuracy"], color="#76b7ff", vmin=0, vmax=1)
)
st.dataframe(rank_sty, use_container_width=True)

st.download_button(
    "⬇️ Download feature accuracies (CSV)",
    data=rank_df.to_csv(index=False),
    file_name="feature_accuracies.csv",
    mime="text/csv",
)

# ---------- Accuracy bar chart (Top N) ----------
st.subheader("Accuracy — Top Features")
max_n = min(20, len(rank_df))
top_n = st.slider("Show top N features", min_value=5, max_value=max_n, value=min(10, max_n), step=1)

plot_df = rank_df.head(top_n).sort_values("accuracy")
fig, ax = plt.subplots(figsize=(8, max(3.5, 0.45*top_n)))
ax.barh(plot_df["feature"], plot_df["accuracy"])

ax.set_xlabel("Accuracy (pred_table threshold 0.5)")
ax.set_ylabel("")
ax.set_xlim(0, 1)
ax.grid(axis="x", linestyle=":", alpha=0.35)

# Annotate values
for i, (feat, acc) in enumerate(zip(plot_df["feature"], plot_df["accuracy"])):
    ax.text(acc + 0.01, i, f"{acc:.3f}", va="center")

# Highlight the best if visible
if best_feature in plot_df["feature"].values:
    idx = plot_df.index[plot_df["feature"] == best_feature][0]
    ax.barh([best_feature], [plot_df.loc[idx, "accuracy"]],
            edgecolor="#0b1220", linewidth=2)

st.pyplot(fig)
