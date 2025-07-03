
# Cloud Kitchen Streamlit Dashboard

This repository hosts an end‑to‑end Streamlit dashboard for exploring a synthetic cloud‑kitchen survey dataset.

## Features

| Tab | What you get |
|-----|--------------|
| **Data Visualisation** | 10+ ready‑made interactive charts plus global filters |
| **Classification** | KNN, Decision Tree, Random Forest & Gradient Boosting with confusion matrices, ROC curves, and CSV prediction downloads |
| **Clustering** | K‑means with dynamic *k*, elbow chart, persona table, and download button |
| **Association Rules** | Apriori‑based market‑basket mining with configurable thresholds |
| **Regression Insights** | Linear, Ridge, Lasso, Decision Tree regressors with error metrics & quick‑hit insights |

## Quickstart (Streamlit Cloud)

1. **Fork this repo** or create a new one and push these files plus your CSV dataset.
2. On [Streamlit Cloud](https://share.streamlit.io):
   - Click *New app* → select your repo
   - Set **Main file** to `apps.py`
   - Deploy 🚀
3. To use your own data: update the GitHub raw URL in the sidebar **or** upload a CSV on the fly.

## Local run

```bash
pip install -r requirements.txt
streamlit run apps.py
```

## Dataset

The default raw link in `apps.py` expects `cloud_kitchen_survey_synthetic.csv` to be in the repo root. Replace the link or upload a new file via the app sidebar.

---

© 2025, your‑name‑here
