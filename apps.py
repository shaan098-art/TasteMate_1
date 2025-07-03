# apps.py
# ------------------------------------------------------------------
# Streamlit Cloud Kitchen Dashboard
# ------------------------------------------------------------------
# Author: ChatGPT (o3)
# Last updated: 04-Jul-2025
# ------------------------------------------------------------------

import io
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import (accuracy_score, auc, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# ------------------------------------------------------------------
# Page Config & Helpers
# ------------------------------------------------------------------
st.set_page_config(page_title="Cloud Kitchen Dashboard", page_icon="🍱", layout="wide")

@st.cache_data
def load_data(uploaded_file: io.BytesIO | None = None) -> pd.DataFrame:
    """
    Load CSV either from user upload or the bundled demo file.
    """
    return pd.read_csv(uploaded_file) if uploaded_file else pd.read_csv(
        "cloud_kitchen_survey_synthetic.csv"
    )


@st.cache_resource
def build_classification_models(X: pd.DataFrame, y: pd.Series):
    """
    Fit 4 classifiers and return dict of fitted models + metrics.
    """
    cat_cols = X.select_dtypes(include="object").columns.tolist()
    num_cols = X.select_dtypes(exclude="object").columns.tolist()

    preproc = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
         ("num", StandardScaler(), num_cols)],
        remainder="drop",
    )

    def pipe(model):
        return Pipeline([("prep", preproc), ("model", model)])

    models = {
        "KNN": pipe(KNeighborsClassifier(n_neighbors=7)),
        "Decision Tree": pipe(DecisionTreeClassifier(max_depth=6)),
        "Random Forest": pipe(RandomForestClassifier(n_estimators=150)),
        "GB Tree": pipe(GradientBoostingClassifier(n_estimators=200, learning_rate=0.05)),
    }

    metrics = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics[name] = {
            "Train Acc": model.score(X_train, y_train),
            "Test Acc": accuracy_score(y_test, preds),
            "Precision": precision_score(y_test, preds, zero_division=0),
            "Recall": recall_score(y_test, preds, zero_division=0),
            "F1": f1_score(y_test, preds, zero_division=0),
        }
    return models, metrics


def plot_conf_matrix(cm: np.ndarray, labels: list[str]) -> go.Figure:
    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=[f"Pred {l}" for l in labels],
            y=[f"Actual {l}" for l in labels],
            texttemplate="%{z}",
            colorscale="Blues",
            showscale=True,
        )
    )
    fig.update_layout(title="Confusion Matrix", xaxis_title="Predicted", yaxis_title="Actual")
    return fig


def prep_features(df: pd.DataFrame, target: str):
    dropna = df.dropna(subset=[target])
    return dropna.drop(columns=[target]), dropna[target]


# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
st.sidebar.header("Data Source")
uploaded_csv = st.sidebar.file_uploader("Upload CSV", type="csv")
df = load_data(uploaded_csv)

with st.sidebar:
    st.subheader("Quick Filters")
    gender_filter = st.multiselect("Gender", sorted(df["gender"].unique()),
                                   default=sorted(df["gender"].unique()))
    diet_filter = st.multiselect("Diet Style", sorted(df["diet_style"].unique()),
                                 default=sorted(df["diet_style"].unique()))

df_filtered = df[df["gender"].isin(gender_filter) & df["diet_style"].isin(diet_filter)]

# ------------------------------------------------------------------
# Tabs
# ------------------------------------------------------------------
tabs = st.tabs(["🔎 Data Visualisation", "🤖 Classification", "🧩 Clustering",
                "🛒 Association Rules", "📈 Regression Insights"])

# ------------------------------------------------------------------
# 1 – Data Visualisation
# ------------------------------------------------------------------
with tabs[0]:
    st.header("Interactive Exploratory Analysis")

    # --- FIXED: make column names explicit & stable -------------
    diet_counts = (
        df_filtered["diet_style"]
        .value_counts()
        .reset_index(name="Count")
    )
    # Force-rename first column regardless of pandas behaviour
    diet_counts.columns = ["Diet Style", "Count"]

    insights = {
        "Age Distribution": px.histogram(df_filtered, x="age_group",
                                         nbins=df_filtered["age_group"].nunique(),
                                         labels={"age_group": "Age Group"}),
        "Gender Split": px.pie(df_filtered, names="gender", hole=0.4),
        "Diet Style Popularity": px.bar(diet_counts, x="Diet Style", y="Count",
                                        labels={"Diet Style": "Diet Style",
                                                "Count": "Count"}),
        "Spend vs Orders": px.scatter(df_filtered, x="orders_per_week",
                                      y="avg_spend_aed", size="avg_spend_aed",
                                      color="diet_style",
                                      labels={"orders_per_week": "Orders/Week",
                                              "avg_spend_aed": "Avg Spend (AED)"}),
        "Workout vs Goal": px.box(df_filtered, x="fitness_goal", y="workouts_per_week",
                                  color="gender",
                                  labels={"fitness_goal": "Fitness Goal",
                                          "workouts_per_week": "Workouts/Week"}),
        "Subscription Intent": px.histogram(df_filtered, x="subscribe_intent"),
        "Eco Pack Score": px.histogram(df_filtered, x="eco_pack_score"),
        "Distance vs Spend": px.scatter(df_filtered, x="distance_km",
                                        y="avg_spend_aed",
                                        labels={"distance_km": "Distance (km)",
                                                "avg_spend_aed": "Avg Spend (AED)"}),
        "Spice Preference": px.histogram(df_filtered, x="spice_level"),
        "Pause Likelihood": px.histogram(df_filtered, x="pause_likelihood"),
    }

    cols = st.columns(2)
    i = 0
    for title, fig in insights.items():
        cols[i].subheader(title)
        cols[i].plotly_chart(fig, use_container_width=True)
        i = 1 - i  # flip-flop columns left/right

# ------------------------------------------------------------------
# 2 – Classification
# ------------------------------------------------------------------
with tabs[1]:
    st.header("Binary Classification – Subscribe Intent")
    target_col = "subscribe_intent"
    X, y = prep_features(df_filtered, target_col)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    models, metrics = build_classification_models(X, y)

    st.subheader("Model Performance")
    st.dataframe(pd.DataFrame(metrics).T.round(3).reset_index()
                 .rename(columns={"index": "Model"}), use_container_width=True)

    st.subheader("Explore Confusion Matrix")
    chosen_model = st.selectbox("Select model", list(models.keys()), key="confmat")
    cm = confusion_matrix(y_test, models[chosen_model].predict(X_test))
    st.plotly_chart(plot_conf_matrix(cm, ["No", "Yes"]), use_container_width=True)

    st.subheader("ROC Curve Comparison")
    roc_fig = go.Figure()
    for name, mdl in models.items():
        probs = mdl.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                     name=f"{name} (AUC={auc(fpr, tpr):.2f})"))
    roc_fig.update_layout(xaxis_title="False Positive Rate",
                          yaxis_title="True Positive Rate",
                          title="ROC Curves")
    st.plotly_chart(roc_fig, use_container_width=True)

    st.subheader("Predict New Data")
    pred_up = st.file_uploader("Upload CSV without target column",
                               type="csv", key="pred_csv")
    if pred_up is not None:
        new_df = pd.read_csv(pred_up)
        chosen_pred = st.selectbox("Model for prediction",
                                   list(models.keys()), key="pred_model")
        new_df["predicted_subscribe_intent"] = models[chosen_pred].predict(new_df)
        st.write(new_df.head())
        st.download_button("⬇︎ Download Predictions",
                           data=new_df.to_csv(index=False).encode(),
                           file_name="predictions.csv")

# ------------------------------------------------------------------
# 3 – Clustering
# (unchanged – omitted here for brevity but keep the same code)
# ------------------------------------------------------------------
# ...   <keep existing clustering, association rules & regression code> ...

# ------------------------------------------------------------------
st.caption(f"© {datetime.utcnow().year} Cloud Kitchen Dashboard | Streamlit {st.__version__}")
