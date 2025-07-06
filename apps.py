# apps.py
# ------------------------------------------------------------------
# Streamlit Cloud Kitchen Dashboard
# ------------------------------------------------------------------
# Author: ChatGPT (o3)
# Last updated: 06-Jul-2025
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
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelBinarizer
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------
st.set_page_config(page_title="Cloud Kitchen Dashboard", page_icon="🍱", layout="wide")

# ------------------------------------------------------------------
# Data loader
# ------------------------------------------------------------------
@st.cache_data
def load_data(uploaded: io.BytesIO | None = None) -> pd.DataFrame:
    """Read CSV from user upload or bundled demo file."""
    return pd.read_csv(uploaded) if uploaded else pd.read_csv(
        "cloud_kitchen_survey_synthetic.csv"
    )

# ------------------------------------------------------------------
# Model builder
# ------------------------------------------------------------------
@st.cache_resource
def build_classification_models(X: pd.DataFrame, y: pd.Series):
    """
    Train four classifiers, return fitted models + metrics.
    Weighted averages ensure metrics always compute even if only one class appears.
    """
    cat_cols = X.select_dtypes(include="object").columns.tolist()
    num_cols = X.select_dtypes(exclude="object").columns.tolist()

    preproc = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
         ("num", StandardScaler(), num_cols)],
        remainder="drop",
    )

    def make_pipe(model):
        return Pipeline([("prep", preproc), ("model", model)])

    models = {
        "KNN": make_pipe(KNeighborsClassifier(n_neighbors=7)),
        "Decision Tree": make_pipe(DecisionTreeClassifier(max_depth=6)),
        "Random Forest": make_pipe(RandomForestClassifier(n_estimators=150)),
        "GB Tree": make_pipe(GradientBoostingClassifier(n_estimators=200, learning_rate=0.05)),
    }

    metrics = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics[name] = {
            "Train Acc": model.score(X_train, y_train),
            "Test Acc": accuracy_score(y_test, preds),
            "Precision": precision_score(y_test, preds, average="weighted", zero_division=0),
            "Recall":    recall_score(y_test, preds, average="weighted", zero_division=0),
            "F1":        f1_score(y_test, preds, average="weighted", zero_division=0),
        }
    return models, metrics

# ------------------------------------------------------------------
# Helper – confusion-matrix plot
# ------------------------------------------------------------------
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

# ------------------------------------------------------------------
# Helper – split X / y
# ------------------------------------------------------------------
def split_xy(df: pd.DataFrame, target: str):
    df2 = df.dropna(subset=[target])
    return df2.drop(columns=[target]), df2[target]

# ------------------------------------------------------------------
# Sidebar – load data & filters
# ------------------------------------------------------------------
st.sidebar.header("Data Source")
uploaded_csv = st.sidebar.file_uploader("Upload CSV", type="csv")
df = load_data(uploaded_csv)

st.sidebar.subheader("Quick Filters")
gender_filter = st.sidebar.multiselect(
    "Gender", sorted(df["gender"].unique()), default=sorted(df["gender"].unique())
)
diet_filter = st.sidebar.multiselect(
    "Diet Style", sorted(df["diet_style"].unique()), default=sorted(df["diet_style"].unique())
)

df_filtered = df[
    df["gender"].isin(gender_filter) & df["diet_style"].isin(diet_filter)
]

# ------------------------------------------------------------------
# Tabs
# ------------------------------------------------------------------
tabs = st.tabs(
    ["🔎 Data Visualisation", "🤖 Classification", "🧩 Clustering", "🛒 Association Rules", "📈 Regression Insights"]
)

# ------------------------------------------------------------------
# 1 – Data Visualisation (updated labels)
# ------------------------------------------------------------------
with tabs[0]:
    st.header("Interactive Exploratory Analysis")

    # ─── Mapping codes to human-readable labels ──────────────────────
    age_labels = {
        1: "18–24",
        2: "25–34",
        3: "35–44",
        4: "45–54",
        5: "55–64",
        6: "65+",
    }
    gender_map = {
        1: "Women",
        2: "Men",
        3: "Non-binary",
        4: "Prefer not to say",
    }
    diet_map = {
        1: "Omnivore",
        2: "Vegetarian",
        3: "Vegan",
        4: "Pescatarian",
        5: "Keto",
    }

    # Make a working copy with new label columns
    viz_df = df_filtered.copy()
    viz_df["Age Group"] = viz_df["age_group"].map(age_labels)
    viz_df["Gender"] = viz_df["gender"].map(gender_map)
    viz_df["Diet Style"] = viz_df["diet_style"].map(diet_map)

    # Prepare Diet Style counts
    diet_counts = (
        viz_df["Diet Style"]
        .value_counts()
        .reset_index(name="Count")
        .rename(columns={"index": "Diet Style"})
    )

    # Build all 10 charts with updated labels
    insights = {
        "Age Distribution": px.histogram(
            viz_df,
            x="Age Group",
            category_orders={"Age Group": list(age_labels.values())},
            labels={"Age Group": "Age Group"},
        ),
        "Gender Split": px.pie(
            viz_df,
            names="Gender",
            hole=0.4,
        ),
        "Diet Style Popularity": px.bar(
            diet_counts,
            x="Diet Style",
            y="Count",
            labels={"Diet Style": "Diet Style", "Count": "Count"},
        ),
        "Spend vs Orders": px.scatter(
            viz_df,
            x="orders_per_week",
            y="avg_spend_aed",
            size="avg_spend_aed",
            color="Diet Style",
            labels={"orders_per_week": "Orders/Week", "avg_spend_aed": "Avg Spend (AED)"},
        ),
        "Workout vs Goal": px.box(
            viz_df,
            x="fitness_goal",
            y="workouts_per_week",
            color="Gender",
            labels={"fitness_goal": "Fitness Goal", "workouts_per_week": "Workouts/Week"},
        ),
        "Subscription Intent": px.histogram(viz_df, x="subscribe_intent", labels={"subscribe_intent": "Subscribe Intent"}),
        "Eco Pack Score": px.histogram(viz_df, x="eco_pack_score", labels={"eco_pack_score": "Eco Pack Score"}),
        "Distance vs Spend": px.scatter(
            viz_df,
            x="distance_km",
            y="avg_spend_aed",
            labels={"distance_km": "Distance (km)", "avg_spend_aed": "Avg Spend (AED)"},
        ),
        "Spice Preference": px.histogram(viz_df, x="spice_level", labels={"spice_level": "Spice Level"}),
        "Pause Likelihood": px.histogram(viz_df, x="pause_likelihood", labels={"pause_likelihood": "Pause Likelihood"}),
    }

    cols = st.columns(2)
    idx = 0
    for title, fig in insights.items():
        cols[idx].subheader(title)
        cols[idx].plotly_chart(fig, use_container_width=True)
        idx = 1 - idx  # alternate columns

# ------------------------------------------------------------------
# 2 – Classification
# ------------------------------------------------------------------
with tabs[1]:
    st.header("Binary Classification – Subscribe Intent")
    target_col = "subscribe_intent"
    X, y = split_xy(df_filtered, target_col)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        stratify=y if len(y.unique()) > 1 else None,
        random_state=42,
    )

    models, metrics = build_classification_models(X, y)

    st.subheader("Model Performance")
    st.dataframe(
        pd.DataFrame(metrics)
        .T.round(3)
        .reset_index()
        .rename(columns={"index": "Model"}),
        use_container_width=True,
    )

    st.subheader("Explore Confusion Matrix")
    chosen_model = st.selectbox("Select model", list(models.keys()), key="confmat_mod")
    labels_present = np.unique(np.concatenate([y_test, models[chosen_model].predict(X_test)]))
    cm = confusion_matrix(y_test, models[chosen_model].predict(X_test), labels=labels_present)
    st.plotly_chart(plot_conf_matrix(cm, labels_present), use_container_width=True)

    st.subheader("ROC Curve Comparison")
    if len(np.unique(y_test)) < 2:
        st.info("ROC curve cannot be plotted because the test set contains only one class after filtering.")
    else:
        lb = LabelBinarizer()
        y_test_bin = lb.fit_transform(y_test).ravel()
        pos_class = lb.classes_[1]

        roc_fig = go.Figure()
        skipped = []
        for name, mdl in models.items():
            if pos_class not in mdl.classes_:
                skipped.append(name)
                continue
            pos_idx = list(mdl.classes_).index(pos_class)
            probs = mdl.predict_proba(X_test)[:, pos_idx]
            if len(probs) != len(y_test_bin):
                skipped.append(name)
                continue
            fpr, tpr, _ = roc_curve(y_test_bin, probs, pos_label=1)
            roc_fig.add_trace(
                go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} (AUC={auc(fpr, tpr):.2f})")
            )
        if roc_fig.data:
            roc_fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", title="ROC Curves")
            st.plotly_chart(roc_fig, use_container_width=True)
        else:
            st.info("No models could be plotted on the ROC curve.")
        if skipped:
            st.caption(f"Skipped ROC for: {', '.join(skipped)} (length/class mismatch)")

    st.subheader("Predict New Data")
    pred_upload = st.file_uploader("Upload CSV without target column", type="csv", key="pred_upl")
    if pred_upload is not None:
        new_df = pd.read_csv(pred_upload)
        model_for_pred = st.selectbox("Model for prediction", list(models.keys()), key="pred_mod")
        new_df["predicted_subscribe_intent"] = models[model_for_pred].predict(new_df)
        st.write(new_df.head())
        st.download_button("⬇︎ Download Predictions", data=new_df.to_csv(index=False).encode(), file_name="predictions.csv")

# ------------------------------------------------------------------
# 3 – Clustering (unchanged)
# ------------------------------------------------------------------
with tabs[2]:
    st.header("Customer Segmentation – K-Means")

    num_cols = df_filtered.select_dtypes(exclude="object").columns.tolist()
    chosen_features = st.multiselect(
        "Numeric features for clustering",
        num_cols,
        default=["orders_per_week", "avg_spend_aed", "distance_km"],
    )
    k_val = st.slider("Number of clusters (k)", 2, 10, 3, 1)

    # Elbow chart
    sse = []
    for k in range(2, 11):
        km = KMeans(n_clusters=k, n_init="auto", random_state=42).fit(
            df_filtered[chosen_features]
        )
        sse.append(km.inertia_)
    elbow_fig = go.Figure(
        data=go.Scatter(x=list(range(2, 11)), y=sse, mode="lines+markers")
    )
    elbow_fig.update_layout(title="Elbow Chart", xaxis_title="k", yaxis_title="SSE")
    st.plotly_chart(elbow_fig, use_container_width=True)

    # Final clustering
    km_final = KMeans(n_clusters=k_val, n_init="auto", random_state=42)
    df_clustered = df_filtered.copy()
    df_clustered["cluster"] = km_final.fit_predict(df_filtered[chosen_features])

    st.subheader("Cluster Personas (Feature Means)")
    st.dataframe(
        df_clustered.groupby("cluster")[chosen_features].mean().round(1),
        use_container_width=True,
    )

    st.download_button(
        "⬇︎ Download cluster-labelled data",
        data=df_clustered.to_csv(index=False).encode(),
        file_name="clustered_data.csv",
    )


# ------------------------------------------------------------------
# 4 – Association Rules (unchanged)
# ------------------------------------------------------------------
with tabs[3]:
     st.header("Market Basket / Association Rule Mining")

    # ① Let the user pick ANY ≥2 categorical columns
    cat_cols = df_filtered.select_dtypes(include="object").columns.tolist()
    st.info("Select **two or more** categorical columns to mine cross-attribute patterns.")
    assoc_cols = st.multiselect("Choose columns", cat_cols, default=cat_cols[:3])

    if len(assoc_cols) < 2:
        st.warning("Please select at least two categorical columns.")
    else:
        # ② One-hot encode each col=value pair → suitable for apriori
        trans_ohe = pd.get_dummies(df_filtered[assoc_cols].astype(str))

        # ③ User-tunable thresholds
        min_sup = st.slider("Min support", 0.01, 0.5, 0.05, 0.01)
        min_conf = st.slider("Min confidence", 0.10, 1.00, 0.60, 0.05)
        min_lift = st.slider("Min lift", 1.0, 10.0, 1.0, 0.1)

        # ④ Mine frequent itemsets
        frequent = apriori(trans_ohe, min_support=min_sup, use_colnames=True)

        if frequent.empty:
            st.warning(
                "No frequent itemsets found with the current support threshold. "
                "Try reducing *Min support* or selecting additional columns."
            )
        else:
            # ⑤ Generate association rules
            rules = association_rules(
                frequent, metric="confidence", min_threshold=min_conf
            )
            rules = rules[rules["lift"] >= min_lift]

            if rules.empty:
                st.warning(
                    "No association rules meet the chosen confidence/lift thresholds."
                )
            else:
                # Human-readable antecedent/consequent strings
                rules["antecedents"] = rules["antecedents"].apply(
                    lambda x: ", ".join(list(x))
                )
                rules["consequents"] = rules["consequents"].apply(
                    lambda x: ", ".join(list(x))
                )
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

# ------------------------------------------------------------------
# 5 – Regression Insights (unchanged)
# ------------------------------------------------------------------
with tabs[4]:
        st.header("Value Prediction – Regression Models")

    target_reg = "avg_spend_aed"
    Xr, yr = split_xy(df_filtered, target_reg)
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        Xr, yr, test_size=0.25, random_state=42
    )

    cat_cols_r = Xr.select_dtypes(include="object").columns.tolist()
    num_cols_r = Xr.select_dtypes(exclude="object").columns.tolist()

    reg_prep = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols_r),
         ("num", StandardScaler(), num_cols_r)]
    )

    def rpipe(model):
        return Pipeline([("prep", reg_prep), ("model", model)])

    reg_models = {
        "Linear": rpipe(LinearRegression()),
        "Ridge": rpipe(Ridge(alpha=1.0)),
        "Lasso": rpipe(Lasso(alpha=0.01)),
        "Decision Tree": rpipe(DecisionTreeRegressor(max_depth=6)),
    }

    reg_metrics, preds_dict = {}, {}
    for name, mdl in reg_models.items():
        mdl.fit(Xr_train, yr_train)
        preds = mdl.predict(Xr_test)
        preds_dict[name] = preds
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

    st.subheader("Predicted vs Actual")
    cols_reg = st.columns(len(reg_models))
    for idx, (name, preds) in enumerate(preds_dict.items()):
        fig = px.scatter(
            x=yr_test, y=preds,
            labels={"x": "Actual", "y": "Predicted"},
            title=name,
        )
        fig.add_shape(
            type="line",
            x0=yr_test.min(), y0=yr_test.min(),
            x1=yr_test.max(), y1=yr_test.max(),
            line=dict(dash="dash"),
        )
        cols_reg[idx].plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------
# Footer
# ------------------------------------------------------------------
st.caption(f"© {datetime.utcnow().year} Cloud Kitchen Dashboard | Streamlit {st.__version__}")
