# apps.py
# ------------------------------------------------------------------
# Streamlit Cloud Kitchen Dashboard
# ------------------------------------------------------------------
# Author: ChatGPT (o3)
# Last updated: 04-Jul-2025
# ------------------------------------------------------------------

import io
import zipfile
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# ------------------------------------------------------------------
# Page Config & Helpers
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Cloud Kitchen Dashboard",
    page_icon="🍱",
    layout="wide",
)

@st.cache_data
def load_data(uploaded_file: io.BytesIO | None = None) -> pd.DataFrame:
    """
    Load CSV either from user upload or from bundled demo file.
    """
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_csv("cloud_kitchen_survey_synthetic.csv")
    return df


@st.cache_resource
def build_classification_models(
    X: pd.DataFrame, y: pd.Series
) -> tuple[dict, dict]:
    """
    Fit 4 classifiers and return dict of fitted models + metrics.
    """
    # Identify column types
    cat_cols = X.select_dtypes(include="object").columns.tolist()
    num_cols = X.select_dtypes(exclude="object").columns.tolist()

    preproc = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", StandardScaler(), num_cols),
        ],
        remainder="drop",
    )

    def make_pipeline(model):
        return Pipeline(steps=[("prep", preproc), ("model", model)])

    models = {
        "KNN": make_pipeline(KNeighborsClassifier(n_neighbors=7)),
        "Decision Tree": make_pipeline(DecisionTreeClassifier(max_depth=6)),
        "Random Forest": make_pipeline(RandomForestClassifier(n_estimators=150)),
        "GB Tree": make_pipeline(
            GradientBoostingClassifier(n_estimators=200, learning_rate=0.05)
        ),
    }

    metrics = {}
    for name, model in models.items():
        model.fit(X, y)
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
    """
    Plotly heat-map style confusion matrix.
    """
    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=[f"Pred {l}" for l in labels],
            y=[f"Actual {l}" for l in labels],
            showscale=True,
            colorscale="Blues",
            texttemplate="%{z}",
        )
    )
    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="Actual",
    )
    return fig


def prep_features(
    df: pd.DataFrame, target: str
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separate X & y, drop rows with missing target.
    """
    dropna = df.dropna(subset=[target])
    y = dropna[target]
    X = dropna.drop(columns=[target])
    return X, y


# ------------------------------------------------------------------
# Sidebar – Data Controls
# ------------------------------------------------------------------
st.sidebar.header("Data Source")
uploaded_csv = st.sidebar.file_uploader("Upload CSV", type="csv")
df = load_data(uploaded_csv)

# Simple high-level filters
with st.sidebar:
    st.subheader("Quick Filters")
    gender_filter = st.multiselect(
        "Gender", sorted(df["gender"].unique()), default=sorted(df["gender"].unique())
    )
    diet_filter = st.multiselect(
        "Diet Style", sorted(df["diet_style"].unique()), default=sorted(df["diet_style"].unique())
    )

df_filtered = df[df["gender"].isin(gender_filter) & df["diet_style"].isin(diet_filter)]

# ------------------------------------------------------------------
# Main Tabs
# ------------------------------------------------------------------
tabs = st.tabs(
    [
        "🔎 Data Visualisation",
        "🤖 Classification",
        "🧩 Clustering",
        "🛒 Association Rules",
        "📈 Regression Insights",
    ]
)

# ------------------------------------------------------------------
# 1 – Data Visualisation
# ------------------------------------------------------------------
with tabs[0]:
    st.header("Interactive Exploratory Analysis")

    # ----------------------------- 10 Descriptive Insights ----------
    diet_counts = (
        df_filtered["diet_style"]
        .value_counts()
        .reset_index(name="Count")
        .rename(columns={"index": "Diet Style"})
    )

    insights = {
        "Age Distribution": px.histogram(
            df_filtered,
            x="age_group",
            nbins=df_filtered["age_group"].nunique(),
            labels={"age_group": "Age Group"},
        ),
        "Gender Split": px.pie(df_filtered, names="gender", hole=0.4),
        "Diet Style Popularity": px.bar(
            diet_counts,
            x="Diet Style",
            y="Count",
            labels={"Diet Style": "Diet Style", "Count": "Count"},
        ),
        "Spend vs Orders": px.scatter(
            df_filtered,
            x="orders_per_week",
            y="avg_spend_aed",
            size="avg_spend_aed",
            color="diet_style",
            labels={"orders_per_week": "Orders/Week", "avg_spend_aed": "Avg Spend (AED)"},
        ),
        "Workout vs Goal": px.box(
            df_filtered,
            x="fitness_goal",
            y="workouts_per_week",
            color="gender",
            labels={
                "fitness_goal": "Fitness Goal",
                "workouts_per_week": "Workouts/Week",
            },
        ),
        "Subscription Intent": px.histogram(df_filtered, x="subscribe_intent"),
        "Eco Pack Score": px.histogram(df_filtered, x="eco_pack_score"),
        "Distance vs Spend": px.scatter(
            df_filtered,
            x="distance_km",
            y="avg_spend_aed",
            labels={"distance_km": "Distance (km)", "avg_spend_aed": "Avg Spend (AED)"},
        ),
        "Spice Preference": px.histogram(df_filtered, x="spice_level"),
        "Pause Likelihood": px.histogram(df_filtered, x="pause_likelihood"),
    }

    # Render charts in two columns
    cols = st.columns(2)
    col_idx = 0
    for title, fig in insights.items():
        cols[col_idx].subheader(title)
        cols[col_idx].plotly_chart(fig, use_container_width=True)
        col_idx = 1 - col_idx  # alternate

# ------------------------------------------------------------------
# 2 – Classification
# ------------------------------------------------------------------
with tabs[1]:
    st.header("Binary Classification – Subscribe Intent")
    target_col = "subscribe_intent"

    # Split
    X, y = prep_features(df_filtered, target_col)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    # Train models + metrics
    models, metrics = build_classification_models(X, y)

    # ---------------- Show aggregated metrics table ---------------
    st.subheader("Model Performance")
    metrics_df = (
        pd.DataFrame(metrics)
        .T.reset_index()
        .rename(columns={"index": "Model"})
        .round(3)
    )
    st.dataframe(metrics_df, use_container_width=True)

    # ---------------- Confusion Matrix toggle --------------------
    st.subheader("Explore Confusion Matrix")
    chosen_model_name = st.selectbox(
        "Select model", list(models.keys()), index=0, key="confmat_selector"
    )
    cm = confusion_matrix(y_test, models[chosen_model_name].predict(X_test))
    st.plotly_chart(
        plot_conf_matrix(cm, labels=["No", "Yes"]), use_container_width=True
    )

    # ---------------- ROC Curve ----------------------------------
    st.subheader("ROC Curve Comparison")
    roc_fig = go.Figure()
    for name, mdl in models.items():
        if hasattr(mdl, "predict_proba"):
            probs = mdl.predict_proba(X_test)[:, 1]
        else:  # e.g. KNN without proba? Actually KNN has predict_proba
            probs = mdl.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"{name} (AUC={auc(fpr, tpr):.2f})",
            )
        )
    roc_fig.update_layout(
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        title="ROC Curves",
    )
    st.plotly_chart(roc_fig, use_container_width=True)

    # ---------------- Upload new data for prediction -------------
    st.subheader("Predict New Data")
    pred_upload = st.file_uploader(
        "Upload CSV without target column to predict ‘subscribe_intent’",
        type="csv",
        key="predict_uploader",
    )
    if pred_upload is not None:
        new_df = pd.read_csv(pred_upload)
        chosen_pred_model = st.selectbox(
            "Choose model for prediction",
            list(models.keys()),
            index=0,
            key="pred_model_selector",
        )
        preds = models[chosen_pred_model].predict(new_df)
        new_df["predicted_subscribe_intent"] = preds
        st.write(new_df.head())

        csv_bytes = new_df.to_csv(index=False).encode()
        st.download_button(
            "⬇︎ Download Predictions", data=csv_bytes, file_name="predictions.csv"
        )

# ------------------------------------------------------------------
# 3 – Clustering
# ------------------------------------------------------------------
with tabs[2]:
    st.header("Customer Segmentation – K-Means")

    # Choose numeric features for clustering (simple heuristic)
    numeric_cols = df_filtered.select_dtypes(exclude="object").columns.tolist()
    selected_features = st.multiselect(
        "Select numeric features for clustering",
        numeric_cols,
        default=["orders_per_week", "avg_spend_aed", "distance_km"],
    )

    cluster_slider = st.slider("Number of clusters (k)", 2, 10, 3, 1)
    kmeans = KMeans(n_clusters=cluster_slider, n_init="auto", random_state=42)

    # Elbow calculation
    sse = []
    for k in range(2, 11):
        km = KMeans(n_clusters=k, n_init="auto", random_state=42).fit(
            df_filtered[selected_features]
        )
        sse.append(km.inertia_)
    elbow_fig = go.Figure(
        data=go.Scatter(x=list(range(2, 11)), y=sse, mode="lines+markers")
    )
    elbow_fig.update_layout(
        title="Elbow Chart", xaxis_title="k", yaxis_title="SSE / Inertia"
    )
    st.plotly_chart(elbow_fig, use_container_width=True)

    # Fit final K-means & label data
    clusters = kmeans.fit_predict(df_filtered[selected_features])
    df_clustered = df_filtered.copy()
    df_clustered["cluster"] = clusters

    # Persona table – averages
    persona = df_clustered.groupby("cluster")[selected_features].mean().round(1)
    st.subheader("Cluster Personas")
    st.dataframe(persona, use_container_width=True)

    # Download full labelled data
    cluster_csv = df_clustered.to_csv(index=False).encode()
    st.download_button(
        "⬇︎ Download labelled data",
        data=cluster_csv,
        file_name="clustered_data.csv",
    )

# ------------------------------------------------------------------
# 4 – Association Rules
# ------------------------------------------------------------------
with tabs[3]:
    st.header("Market Basket / Association Rule Mining")

    candidate_cols = df_filtered.select_dtypes(include="object").columns.tolist()
    st.info(
        "Select exactly **two** categorical columns to build transactions (e.g., "
        "`diet_style` + `fitness_goal`)."
    )
    cols_choice = st.multiselect(
        "Select two columns", candidate_cols, default=candidate_cols[:2]
    )

    if len(cols_choice) == 2:
        # Build transaction list
        trans_df = (
            df_filtered[cols_choice]
            .astype(str)
            .apply(lambda x: x[0] + "_" + x[1], axis=1)
            .to_frame(name="item")
        )
        basket = (
            trans_df.groupby("item")["item"]
            .count()
            .reset_index(name="count")
            .rename(columns={"item": "itemsets"})
        )
        # One-hot encode the item column for apriori
        trans_ohe = trans_df["item"].str.get_dummies()

        min_sup = st.slider("Min support", 0.01, 0.5, 0.05, 0.01)
        min_conf = st.slider("Min confidence", 0.1, 1.0, 0.6, 0.05)
        min_lift = st.slider("Min lift", 1.0, 10.0, 1.0, 0.1)

        frequent = apriori(trans_ohe, min_support=min_sup, use_colnames=True)
        rules = association_rules(frequent, metric="confidence", min_threshold=min_conf)
        rules = rules[rules["lift"] >= min_lift]
        rules = (
            rules.sort_values("confidence", ascending=False)
            .head(10)
            .reset_index(drop=True)
        )

        st.subheader("Top-10 Rules")
        st.dataframe(
            rules[["antecedents", "consequents", "support", "confidence", "lift"]],
            use_container_width=True,
        )
    else:
        st.warning("Please select **exactly two** categorical columns.")

# ------------------------------------------------------------------
# 5 – Regression Insights
# ------------------------------------------------------------------
with tabs[4]:
    st.header("Value Prediction – Regression Models")
    target_reg = "avg_spend_aed"
    X_reg, y_reg = prep_features(df_filtered, target_reg)
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        X_reg, y_reg, test_size=0.25, random_state=42
    )

    # Preprocess
    cat_cols_reg = X_reg.select_dtypes(include="object").columns.tolist()
    num_cols_reg = X_reg.select_dtypes(exclude="object").columns.tolist()

    reg_prep = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols_reg),
            ("num", StandardScaler(), num_cols_reg),
        ]
    )

    def reg_pipe(model):
        return Pipeline([("prep", reg_prep), ("model", model)])

    reg_models = {
        "Linear": reg_pipe(LinearRegression()),
        "Ridge": reg_pipe(Ridge(alpha=1.0)),
        "Lasso": reg_pipe(Lasso(alpha=0.01)),
        "DecisionTree": reg_pipe(DecisionTreeRegressor(max_depth=6)),
    }

    reg_metrics = {}
    preds_vs_actual = {}

    for name, mdl in reg_models.items():
        mdl.fit(Xr_train, yr_train)
        preds = mdl.predict(Xr_test)
        preds_vs_actual[name] = preds
        reg_metrics[name] = {
            "MAE": np.mean(np.abs(yr_test - preds)),
            "MSE": np.mean((yr_test - preds) ** 2),
            "R²": mdl.score(Xr_test, yr_test),
        }

    st.subheader("Model Metrics")
    st.dataframe(
        pd.DataFrame(reg_metrics).T.round(3).reset_index().rename(columns={"index": "Model"}),
        use_container_width=True,
    )

    # 5–7 quick visual insights
    st.subheader("Predicted vs Actual")
    cols_reg = st.columns(len(reg_models))
    for idx, (name, preds) in enumerate(preds_vs_actual.items()):
        fig = px.scatter(
            x=yr_test,
            y=preds,
            labels={"x": "Actual", "y": "Predicted"},
            title=name,
        )
        fig.add_shape(
            type="line",
            x0=yr_test.min(),
            y0=yr_test.min(),
            x1=yr_test.max(),
            y1=yr_test.max(),
            line=dict(dash="dash"),
        )
        cols_reg[idx].plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------
# Footer
# ------------------------------------------------------------------
st.caption(
    f"© {datetime.utcnow().year}  Cloud Kitchen Dashboard | Built with Streamlit {st.__version__}"
)

