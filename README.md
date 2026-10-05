# 💧 Water Quality Intelligence — Streamlit Dashboard

Interactive website version of the supplied Water Quality Analysis project.

## Included analysis

1. Executive Overview
2. Data Understanding
3. Data Cleaning & Preprocessing
4. Univariate Analysis
5. Bivariate Analysis
6. Correlation Analysis
7. IQR Outlier Analysis
8. WHO / Reference Range Analysis
9. Interactive 3D Feature Visualization
10. 3D PCA Visualization
11. Insights & Conclusion
12. Interactive Data Explorer + CSV downloads

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The default dataset expected by the app is:

`water_potability(5).csv`

You can also upload another compatible CSV from the sidebar.

## Important source consistency note

The supplied notebooks contain two different reference-limit dictionaries for the WHO/reference analysis. The dashboard therefore exposes both profiles instead of silently merging them.

The supplied presentation also reports a duplicate-cleaning step, while the uploaded CSV itself currently contains zero duplicate rows. The dashboard calculates these values dynamically from the uploaded data.
