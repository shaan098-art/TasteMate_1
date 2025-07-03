
# -*- coding: utf-8 -*-
"""
Streamlit Cloud Kitchen Dashboard
---------------------------------
Single‑file Streamlit application covering:
1. Data Visualisation (≥10 descriptive insights)
2. Classification (KNN, Decision Tree, Random Forest, Gradient Boosting)
3. Clustering (K‑means with dynamic k)
4. Association Rule Mining (Apriori)
5. Regression Insights (Linear, Ridge, Lasso, Decision Tree Regressor)

The app pulls data from a GitHub raw link by default but also allows users to
upload their own CSVs. All heavy‑lifting lives in this file to simplify deployment
on Streamlit Cloud (no extra modules or folders required).

Author: ChatGPT o3
"""

import streamlit as st
import pandas as pd
import numpy as np
from urllib.error import URLError
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_curve, auc, mean_squared_error, mean_absolute_error, r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.cluster import KMeans
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from mlxtend.frequent_patterns import apriori, association_rules
import io
import base64

st.set_page_config(page_title="Cloud Kitchen Dashboard", layout="wide")

# ---------- Helper functions -------------------------------------------------

@st.cache_data
def load_data(source: str) -> pd.DataFrame:
    """Load data from GitHub raw link or local path"""
    try:
        df = pd.read_csv(source)
    except URLError:
        st.error("Failed loading data from GitHub. Upload a CSV instead.")
        df = pd.DataFrame()
    return df

GITHUB_RAW_URL = "https://raw.githubusercontent.com/<YOUR-USERNAME>/<YOUR-REPO>/main/cloud_kitchen_survey_synthetic.csv"

st.sidebar.header("Data Source")
data_source = st.sidebar.text_input("GitHub raw CSV URL", value=GITHUB_RAW_URL)
uploaded_file = st.sidebar.file_uploader("...or upload your own CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = load_data(data_source)

if df.empty:
    st.stop()

st.sidebar.success(f"Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns.")

# Identify numeric and categorical columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

# ---------- Tab 1: Data Visualisation ---------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Data Visualisation", "🤖 Classification", "🔍 Clustering", "🔗 Association Rules", "📈 Regression Insights"])

with tab1:
    st.header("Exploratory Insights")

    # Global filters
    with st.expander("Add global filters"):
        col1, col2 = st.columns(2)
        with col1:
            age_filter = st.slider("Age Group", int(df['age_group'].min()), int(df['age_group'].max()), (int(df['age_group'].min()), int(df['age_group'].max())))
        with col2:
            spend_filter = st.slider("Average Spend (AED)", float(df['avg_spend_aed'].min()), float(df['avg_spend_aed'].max()), (float(df['avg_spend_aed'].min()), float(df['avg_spend_aed'].max())))

    df_filtered = df[(df['age_group'].between(*age_filter)) & (df['avg_spend_aed'].between(*spend_filter))]

    st.markdown(f"**Filtered dataset:** {df_filtered.shape[0]} rows")

    # Draw 10 insights
    insights = {
        "Age Distribution": px.histogram(df_filtered, x='age_group', nbins=len(df['age_group'].unique()), labels={'age_group':'Age Group'}),
        "Gender Split": px.pie(df_filtered, names='gender', hole=.4),
        "Diet Style Popularity": px.bar(df_filtered['diet_style'].value_counts().reset_index(), x='index', y='diet_style', labels={'index':'Diet Style','diet_style':'Count'}),
        "Spend vs Orders": px.scatter(df_filtered, x='orders_per_week', y='avg_spend_aed', size='avg_spend_aed', color='diet_style', labels={'orders_per_week':'Orders/Week','avg_spend_aed':'Avg Spend (AED)'}),
        "Workout vs Fitness Goal": px.box(df_filtered, x='fitness_goal', y='workouts_per_week', points='all', labels={'fitness_goal':'Fitness Goal','workouts_per_week':'Workouts/Week'}),
        "Subscription Intent": px.histogram(df_filtered, x='subscribe_intent'),
        "Eco Pack Score": px.histogram(df_filtered, x='eco_pack_score'),
        "Distance vs Spend": px.scatter(df_filtered, x='distance_km', y='avg_spend_aed'),
        "Spice Level Preference": px.histogram(df_filtered, x='spice_level'),
        "Pause Likelihood": px.histogram(df_filtered, x='pause_likelihood')
    }

    for title, fig in insights.items():
        st.subheader(title)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"*Insight:* {title} chart reveals distribution and potential patterns.")

# ---------- Tab 2: Classification -------------------------------------------
with tab2:
    st.header("Diet Style Classification")

    target_col = st.selectbox("Choose target variable", options=[col for col in df.columns if df[col].nunique() < 15], index=[col for col in df.columns].index('diet_style') if 'diet_style' in df.columns else 0)
    st.info(f"Current target: **{target_col}**")

    # Prepare data
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ])

    models = {
        'KNN': KNeighborsClassifier(),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42)
    }

    metrics_rows = []
    roc_data = {}

    for name, model in models.items():
        pipe = Pipeline(steps=[('prep', preprocessor), ('model', model)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

        metrics_rows.append({'Model': name, 'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1‑Score': f1})

        # ROC data (One‑vs‑Rest)
        y_test_bin = label_binarize(y_test, classes=np.unique(y))
        y_score = pipe.predict_proba(X_test)
        fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
        roc_auc = auc(fpr, tpr)
        roc_data[name] = (fpr, tpr, roc_auc)

    metrics_df = pd.DataFrame(metrics_rows).sort_values('Accuracy', ascending=False)
    st.subheader("Performance Metrics")
    st.dataframe(metrics_df.style.format({c:"{:.3f}" for c in ['Accuracy','Precision','Recall','F1‑Score']}))

    # Confusion matrix
    st.subheader("Confusion Matrix")
    cm_model = st.selectbox("Select model", list(models.keys()))
    model_idx = [row['Model'] for row in metrics_rows].index(cm_model)
    chosen_model = list(models.values())[model_idx]
    # Re‑fit chosen model
    pipe = Pipeline(steps=[('prep', preprocessor), ('model', chosen_model)])
    pipe.fit(X_train, y_train)
    y_pred_cm = pipe.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_cm)
    fig_cm = px.imshow(cm,
                       text_auto=True,
                       labels=dict(x="Predicted", y="Actual", color="Count"),
                       x=np.unique(y),
                       y=np.unique(y))
    st.plotly_chart(fig_cm, use_container_width=True)

    # ROC curves
    st.subheader("ROC Curves")
    fig_roc = go.Figure()
    for name, (fpr, tpr, roc_auc) in roc_data.items():
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f"{name} (AUC={roc_auc:.2f})"))
    fig_roc.add_shape(type='line', x0=0, y0=0, x1=1, y1=1, line=dict(dash='dash'))
    fig_roc.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", xaxis=dict(scaleanchor="x", scaleratio=1))
    st.plotly_chart(fig_roc, use_container_width=True)

    # Predict on new data
    st.subheader("Predict on New Data")
    new_file = st.file_uploader("Upload CSV without target column", key="predict_upload", type=["csv"])
    if new_file:
        new_df = pd.read_csv(new_file)
        predictions = pipe.predict(new_df)
        output = new_df.copy()
        output[f"predicted_{target_col}"] = predictions
        st.dataframe(output.head())

        csv = output.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        st.download_button(label="Download predictions", data=csv, file_name="predictions.csv", mime="text/csv")

# ---------- Tab 3: Clustering ------------------------------------------------
with tab3:
    st.header("Customer Segmentation (K‑means)")
    cluster_features = st.multiselect("Select features for clustering", numeric_cols, default=['orders_per_week','avg_spend_aed','distance_km'])
    if len(cluster_features) < 2:
        st.warning("Please select at least two features.")
        st.stop()

    standardized = StandardScaler().fit_transform(df[cluster_features])
    k_range = range(2, 11)
    sse = [KMeans(n_clusters=k, random_state=42).fit(standardized).inertia_ for k in k_range]

    fig_elbow = go.Figure()
    fig_elbow.add_trace(go.Scatter(x=list(k_range), y=sse, mode='lines+markers'))
    fig_elbow.update_layout(xaxis_title="k", yaxis_title="Sum of Squared Errors", title="Elbow Chart")
    st.plotly_chart(fig_elbow, use_container_width=True)

    k = st.slider("Number of clusters", 2, 10, 3)
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(standardized)
    df_clusters = df.copy()
    df_clusters['cluster'] = labels

    # Persona table (centroids)
    centroids = pd.DataFrame(kmeans.cluster_centers_, columns=cluster_features)
    centroids.index.name = 'cluster'
    st.subheader("Cluster Personas")
    st.dataframe(centroids)

    # Download labelled data
    csv_clusters = df_clusters.to_csv(index=False)
    st.download_button("Download data with cluster labels", data=csv_clusters, file_name="clustered_data.csv", mime="text/csv")

# ---------- Tab 4: Association Rules ----------------------------------------
with tab4:
    st.header("Market Basket Insights (Apriori)")
    # Convert fav_add_ons to transaction binary matrix
    trans_col = st.selectbox("Select column containing comma‑separated items", ['fav_add_ons'])
    transactions = df[trans_col].fillna('').apply(lambda x: x.split(','))
    all_items = sorted({item.strip() for sublist in transactions for item in sublist if item})
    one_hot = pd.DataFrame(0, index=df.index, columns=all_items)
    for idx, items in transactions.items():
        for item in items:
            item = item.strip()
            if item:
                one_hot.loc[idx, item] = 1

    # Apriori params
    min_support = st.slider("Minimum support", 0.01, 0.5, 0.05, 0.01)
    min_conf = st.slider("Minimum confidence", 0.1, 1.0, 0.3, 0.05)
    min_lift = st.slider("Minimum lift", 1.0, 10.0, 1.0, 0.1)

    freq_items = apriori(one_hot, min_support=min_support, use_colnames=True)
    rules = association_rules(freq_items, metric="confidence", min_threshold=min_conf)
    rules = rules[rules['lift'] >= min_lift]
    rules = rules.sort_values('confidence', ascending=False).head(10)

    st.subheader("Top‑10 Association Rules")
    st.dataframe(rules[['antecedents','consequents','support','confidence','lift']])

# ---------- Tab 5: Regression Insights --------------------------------------
with tab5:
    st.header("Spend Prediction & Insights")

    reg_target = st.selectbox("Select numeric target variable", numeric_cols, index=numeric_cols.index('avg_spend_aed') if 'avg_spend_aed' in numeric_cols else 0)
    st.info(f"Predicting **{reg_target}**")

    X_reg = df.drop(columns=[reg_target])
    y_reg = df[reg_target]

    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

    reg_models = {
        'Linear': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.001),
        'Decision Tree': DecisionTreeRegressor(random_state=42, max_depth=5)
    }

    reg_rows = []
    for name, model in reg_models.items():
        pipe_reg = Pipeline(steps=[('prep', preprocessor), ('model', model)])
        pipe_reg.fit(X_train_reg, y_train_reg)
        preds = pipe_reg.predict(X_test_reg)

        rmse = mean_squared_error(y_test_reg, preds, squared=False)
        mae = mean_absolute_error(y_test_reg, preds)
        r2 = r2_score(y_test_reg, preds)

        reg_rows.append({'Model': name, 'RMSE': rmse, 'MAE': mae, 'R²': r2})

        # Scatter plot for quick insights
        fig_scatter = px.scatter(x=y_test_reg, y=preds, labels={'x':'Actual','y':'Predicted'}, title=f"{name}: Actual vs Predicted")
        st.plotly_chart(fig_scatter, use_container_width=True)

    reg_df = pd.DataFrame(reg_rows).sort_values('RMSE')
    st.subheader("Model Comparison")
    st.dataframe(reg_df.style.format({c:"{:.2f}" for c in ['RMSE','MAE','R²']}))

    st.markdown("""### Quick Insights
    1. Linear models provide a baseline; Ridge regularization can reduce overfitting.
    2. Lasso performs feature selection by shrinking insignificant coefficients to zero.
    3. Decision trees capture non‑linear relationships but risk overfitting; shallow depth mitigates this.
    4. Lower RMSE/MAE indicate better predictive accuracy for spend estimation.
    5. R² close to 1 suggests strong explanatory power of selected features.
    6. Discrepancies between actual and predicted points highlight segments needing separate pricing strategies.
    7. Combining regression predictions with clustering can personalize pricing and promotions.""")
