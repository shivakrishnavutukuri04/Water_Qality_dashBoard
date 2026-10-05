
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Water Quality Intelligence",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #07111f 0%, #0b1728 45%, #06131f 100%);
        color: #eef6ff;
    }
    section[data-testid="stSidebar"] {
        background: #081522;
        border-right: 1px solid #1e3a52;
    }
    .hero {
        padding: 32px;
        border-radius: 22px;
        background: linear-gradient(135deg, #0d2940, #0b1c2c);
        border: 1px solid #24516e;
        margin-bottom: 25px;
    }
    .hero h1 {
        font-size: 42px;
        margin-bottom: 6px;
    }
    .hero p {
        color: #a9c6da;
        font-size: 17px;
    }
    .metric-card {
        background: #0d2030;
        border: 1px solid #21465d;
        border-radius: 15px;
        padding: 18px;
        text-align: center;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #5ed6ff;
    }
    .metric-label {
        color: #9eb8c9;
        font-size: 14px;
    }
    .step {
        padding: 14px 18px;
        border-left: 4px solid #36c5f0;
        background: #0b1c2a;
        border-radius: 8px;
        margin: 8px 0 18px 0;
    }
    .insight {
        padding: 16px 20px;
        border-radius: 12px;
        background: #0d2433;
        border: 1px solid #24516e;
        margin: 10px 0;
    }
    .warning-box {
        padding: 15px;
        border-radius: 10px;
        background: #2b2110;
        border: 1px solid #8a6825;
        color: #f5d58b;
    }
    h2, h3 {
        color: #dff5ff;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# DATA
# ============================================================
DEFAULT_FILE = "water_potability(5).csv"

@st.cache_data
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    return pd.read_csv(DEFAULT_FILE)

uploaded = st.sidebar.file_uploader(
    "Upload Water Potability CSV",
    type=["csv"],
    help="Upload a CSV with the same water-quality columns."
)

try:
    df_raw = load_data(uploaded)
except FileNotFoundError:
    st.error("Default CSV not found. Upload your water_potability CSV from the sidebar.")
    st.stop()

TARGET = "Potability"
FEATURES = [c for c in df_raw.columns if c != TARGET]

# Median-imputed working data, matching the methodology in the presentation.
df = df_raw.copy()
numeric_features = df.select_dtypes(include=np.number).columns.tolist()
for col in [c for c in numeric_features if c != TARGET]:
    df[col] = df[col].fillna(df[col].median())

duplicate_count = df_raw.duplicated().sum()
missing_before = int(df_raw.isna().sum().sum())
missing_after = int(df.isna().sum().sum())

# ============================================================
# HELPERS
# ============================================================
def metric_card(label, value):
    st.markdown(
        f'<div class="metric-card"><div class="metric-value">{value}</div>'
        f'<div class="metric-label">{label}</div></div>',
        unsafe_allow_html=True
    )

def iqr_info(data, col):
    q1 = data[col].quantile(0.25)
    q3 = data[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    mask = (data[col] < lower) | (data[col] > upper)
    return int(mask.sum()), q1, q3, lower, upper

# Reference profiles found in the supplied notebook.
WHO_PROFILES = {
    "Notebook reference profile A": {
        "ph": (6.5, 8.5),
        "Hardness": (0, 600),
        "Solids": (0, 2000),
        "Chloramines": (0, 4),
        "Sulfate": (0, 400),
        "Conductivity": (50, 500),
        "Organic_carbon": (0, 4),
        "Trihalomethanes": (0, 80),
        "Turbidity": (0, 5),
    },
    "Notebook reference profile B": {
        "ph": (6.5, 8.5),
        "Hardness": (0, 600),
        "Solids": (0, 1000),
        "Chloramines": (0, 4),
        "Sulfate": (0, 250),
        "Conductivity": (50, 400),
        "Organic_carbon": (0, 4),
        "Trihalomethanes": (0, 80),
        "Turbidity": (0, 5),
    },
}

# ============================================================
# SIDEBAR NAVIGATION
# ============================================================
st.sidebar.markdown("## 💧 Water Quality")
st.sidebar.caption("Interactive Data Analysis Dashboard")

page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Executive Overview",
        "1️⃣ Data Understanding",
        "2️⃣ Data Cleaning",
        "3️⃣ Univariate Analysis",
        "4️⃣ Bivariate Analysis",
        "5️⃣ Correlation Analysis",
        "6️⃣ Outlier Analysis",
        "7️⃣ WHO Reference Analysis",
        "8️⃣ 3D Analytics",
        "9️⃣ Insights & Conclusion",
        "📋 Interactive Data Explorer",
    ]
)

st.sidebar.divider()
st.sidebar.caption("Source: supplied water quality analysis notebooks + CSV")
st.sidebar.caption("Analysis is interactive; charts recalculate from the uploaded data.")

# ============================================================
# HERO
# ============================================================
st.markdown("""
<div class="hero">
    <h1>💧 Water Quality Intelligence</h1>
    <p>Exploratory analysis of water potability using Python, Pandas and interactive visual analytics.</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# EXECUTIVE OVERVIEW
# ============================================================
if page == "🏠 Executive Overview":
    st.markdown("## Project Overview")
    st.write(
        "The objective is to understand whether water samples are classified as potable "
        "or non-potable and to investigate the chemical and physical variables associated "
        "with that classification."
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Total Samples", f"{len(df_raw):,}")
    with c2: metric_card("Features", len(FEATURES))
    with c3: metric_card("Missing Values", f"{missing_before:,}")
    with c4: metric_card("Duplicate Rows", f"{duplicate_count:,}")

    st.markdown("### Analysis Pipeline")
    steps = [
        ("01", "Data Understanding", "Shape, columns, data types, statistical summary and target distribution."),
        ("02", "Data Cleaning", "Missing-value analysis, median imputation and duplicate verification."),
        ("03", "Univariate Analysis", "Feature distributions, skewness, kurtosis and class balance."),
        ("04", "Bivariate Analysis", "Feature-vs-target comparisons and group-wise statistics."),
        ("05", "Correlation Analysis", "Linear relationships among water-quality variables."),
        ("06", "Outlier Analysis", "IQR-based detection and feature-wise outlier counts."),
        ("07", "WHO Reference Analysis", "Reference-range violation analysis."),
        ("08", "3D Analytics", "Interactive feature-space and PCA visualization."),
        ("09", "Insights", "Key observations, limitations and future scope."),
    ]
    for num, title, desc in steps:
        st.markdown(
            f'<div class="step"><b>{num} — {title}</b><br><span style="color:#a9c6da">{desc}</span></div>',
            unsafe_allow_html=True
        )

    st.markdown("### Potability Balance")
    counts = df[TARGET].value_counts().sort_index()
    fig = px.pie(
        values=counts.values,
        names=["Unsafe (0)", "Safe (1)"],
        hole=0.52,
        title="Water Potability Distribution"
    )
    fig.update_layout(template="plotly_dark", height=430)
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# DATA UNDERSTANDING
# ============================================================
elif page == "1️⃣ Data Understanding":
    st.markdown("## 1. Data Understanding")

    c1, c2, c3 = st.columns(3)
    with c1: metric_card("Rows", f"{df_raw.shape[0]:,}")
    with c2: metric_card("Columns", df_raw.shape[1])
    with c3: metric_card("Target", TARGET)

    st.markdown("### Dataset Preview")
    n = st.slider("Rows to display", 5, 50, 10)
    st.dataframe(df_raw.head(n), use_container_width=True)

    st.markdown("### Column Information")
    info = pd.DataFrame({
        "Feature": df_raw.columns,
        "Data Type": [str(df_raw[c].dtype) for c in df_raw.columns],
        "Missing": [int(df_raw[c].isna().sum()) for c in df_raw.columns],
        "Unique Values": [int(df_raw[c].nunique()) for c in df_raw.columns],
    })
    st.dataframe(info, use_container_width=True, hide_index=True)

    st.markdown("### Feature Dictionary")
    meanings = {
        "ph": "Acidity / alkalinity level",
        "Hardness": "Calcium and magnesium content",
        "Solids": "Total dissolved solids",
        "Chloramines": "Disinfectant residual",
        "Sulfate": "Chemical concentration",
        "Conductivity": "Electrical conductivity",
        "Organic_carbon": "Organic matter content",
        "Trihalomethanes": "Disinfection by-product",
        "Turbidity": "Water cloudiness",
        "Potability": "Target class: 0 = unsafe, 1 = safe",
    }
    dictionary = pd.DataFrame({
        "Feature": FEATURES + [TARGET],
        "Meaning": [meanings.get(c, "Numeric water-quality variable") for c in FEATURES + [TARGET]]
    })
    st.dataframe(dictionary, use_container_width=True, hide_index=True)

    st.markdown("### Statistical Summary")
    st.dataframe(df_raw.describe().T, use_container_width=True)

# ============================================================
# DATA CLEANING
# ============================================================
elif page == "2️⃣ Data Cleaning":
    st.markdown("## 2. Data Cleaning & Preprocessing")

    st.markdown(
        '<div class="step"><b>Cleaning workflow</b><br>'
        'Raw CSV → Missing-value audit → Median imputation → Duplicate check → Analysis-ready dataset</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Raw Missing Values", f"{missing_before:,}")
    with c2: metric_card("Missing After Imputation", f"{missing_after:,}")
    with c3: metric_card("Duplicate Rows", f"{duplicate_count:,}")
    with c4: metric_card("Rows in Analysis", f"{len(df):,}")

    st.markdown("### Missing Values Before Cleaning")
    missing = df_raw.isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0]
    if len(missing):
        fig = px.bar(
            x=missing.index, y=missing.values,
            labels={"x": "Feature", "y": "Missing Values"},
            title="Missing Values by Feature"
        )
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.success("No missing values found in the uploaded data.")

    st.markdown("### Before vs After")
    clean_compare = pd.DataFrame({
        "Feature": df_raw.columns,
        "Before": [int(df_raw[c].isna().sum()) for c in df_raw.columns],
        "After": [int(df[c].isna().sum()) for c in df.columns],
    })
    st.dataframe(clean_compare, use_container_width=True, hide_index=True)

    st.markdown("### Duplicate Check")
    if duplicate_count == 0:
        st.success("No duplicate records were found in the supplied CSV.")
    else:
        st.warning(f"{duplicate_count:,} duplicate records were found.")

    st.download_button(
        "⬇️ Download Analysis-Ready Dataset",
        df.to_csv(index=False).encode("utf-8"),
        "water_quality_cleaned.csv",
        "text/csv"
    )

# ============================================================
# UNIVARIATE
# ============================================================
elif page == "3️⃣ Univariate Analysis":
    st.markdown("## 3. Univariate Analysis")

    counts = df[TARGET].value_counts().sort_index()
    c1, c2 = st.columns(2)
    with c1: metric_card("Unsafe Samples", f"{int(counts.get(0, 0)):,}")
    with c2: metric_card("Safe Samples", f"{int(counts.get(1, 0)):,}")

    st.markdown("### 3.1 Target Distribution")
    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(
            x=["Unsafe (0)", "Safe (1)"],
            y=[counts.get(0, 0), counts.get(1, 0)],
            labels={"x": "Potability", "y": "Count"},
            title="Potability Count"
        )
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.pie(
            values=[counts.get(0, 0), counts.get(1, 0)],
            names=["Unsafe (0)", "Safe (1)"],
            hole=0.45,
            title="Potability Percentage"
        )
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 3.2 Feature Distribution")
    feature = st.selectbox("Select a feature", FEATURES)
    fig = px.histogram(
        df, x=feature, color=TARGET,
        marginal="box", nbins=45,
        barmode="overlay",
        opacity=0.65,
        title=f"{feature} Distribution by Potability"
    )
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 3.3 All Feature Distributions")
    selected_features = st.multiselect(
        "Choose features",
        FEATURES,
        default=FEATURES
    )
    if selected_features:
        long_df = df[selected_features].melt(var_name="Feature", value_name="Value")
        fig = px.histogram(
            long_df, x="Value", facet_col="Feature",
            facet_col_wrap=3, height=900,
            title="Interactive Feature Distributions"
        )
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 3.4 Skewness & Kurtosis")
    stats = pd.DataFrame({
        "Skewness": df[FEATURES].skew(),
        "Kurtosis": df[FEATURES].kurtosis()
    }).sort_values("Skewness", key=abs, ascending=False)
    st.dataframe(stats, use_container_width=True)

    fig = px.bar(
        stats.reset_index(),
        x="index", y="Skewness",
        color="Skewness",
        title="Feature Skewness"
    )
    fig.update_layout(template="plotly_dark", xaxis_title="Feature", xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# BIVARIATE
# ============================================================
elif page == "4️⃣ Bivariate Analysis":
    st.markdown("## 4. Bivariate Analysis")

    feature = st.selectbox("Choose a feature", FEATURES, key="bivar_feature")

    st.markdown("### Feature vs Potability — Box Plot")
    fig = px.box(
        df, x=TARGET, y=feature, color=TARGET,
        points="outliers",
        title=f"{feature} vs Potability"
    )
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Distribution by Potability")
    fig = px.histogram(
        df, x=feature, color=TARGET,
        marginal="violin",
        barmode="overlay",
        opacity=0.60,
        title=f"{feature} Distribution by Potability"
    )
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Group-wise Statistics")
    group_stats = df.groupby(TARGET)[FEATURES].mean().T
    group_stats.columns = ["Unsafe (0)", "Safe (1)"]
    group_stats["Difference (Safe - Unsafe)"] = group_stats["Safe (1)"] - group_stats["Unsafe (0)"]
    st.dataframe(group_stats.sort_values("Difference (Safe - Unsafe)", key=abs, ascending=False),
                 use_container_width=True)

# ============================================================
# CORRELATION
# ============================================================
elif page == "5️⃣ Correlation Analysis":
    st.markdown("## 5. Correlation Analysis")

    corr = df.corr(numeric_only=True)

    fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title="Water Quality Feature Correlation Matrix"
    )
    fig.update_layout(template="plotly_dark", height=750)
    st.plotly_chart(fig, use_container_width=True)

    target_corr = corr[TARGET].drop(TARGET).sort_values()
    corr_table = target_corr.to_frame("Correlation with Potability")
    st.markdown("### Correlation with Potability")
    st.dataframe(corr_table, use_container_width=True)

    fig = px.bar(
        target_corr.reset_index(),
        x="index", y="Potability",
        color="Potability",
        title="Feature Correlation with Potability"
    )
    fig.update_layout(template="plotly_dark", xaxis_title="Feature", xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "The supplied project presentation reports that no feature has a strong linear "
        "relationship with Potability, suggesting that multiple weak signals may act together."
    )

# ============================================================
# OUTLIERS
# ============================================================
elif page == "6️⃣ Outlier Analysis":
    st.markdown("## 6. Outlier Analysis — IQR Method")

    rows = []
    for col in FEATURES:
        count, q1, q3, lower, upper = iqr_info(df, col)
        rows.append({
            "Feature": col,
            "Outliers": count,
            "Q1": q1,
            "Q3": q3,
            "Lower Bound": lower,
            "Upper Bound": upper
        })

    outlier_df = pd.DataFrame(rows).sort_values("Outliers", ascending=False)

    fig = px.bar(
        outlier_df,
        x="Feature", y="Outliers",
        color="Outliers",
        title="Outlier Count by Feature"
    )
    fig.update_layout(template="plotly_dark", xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(outlier_df, use_container_width=True, hide_index=True)

    selected = st.selectbox("Inspect outliers for", FEATURES, key="outlier_feature")
    count, q1, q3, lower, upper = iqr_info(df, selected)

    fig = go.Figure()
    normal = df.loc[~((df[selected] < lower) | (df[selected] > upper))]
    abnormal = df.loc[((df[selected] < lower) | (df[selected] > upper))]

    fig.add_trace(go.Scatter(
        x=normal.index, y=normal[selected],
        mode="markers", name="Normal"
    ))
    fig.add_trace(go.Scatter(
        x=abnormal.index, y=abnormal[selected],
        mode="markers", name="Outlier"
    ))
    fig.add_hline(y=lower, line_dash="dash", annotation_text="Lower IQR bound")
    fig.add_hline(y=upper, line_dash="dash", annotation_text="Upper IQR bound")
    fig.update_layout(
        template="plotly_dark",
        title=f"{selected}: IQR Outlier Detection",
        xaxis_title="Row Index",
        yaxis_title=selected,
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# WHO
# ============================================================
elif page == "7️⃣ WHO Reference Analysis":
    st.markdown("## 7. WHO / Reference Range Violation Analysis")

    st.markdown(
        '<div class="warning-box"><b>Important:</b> The supplied notebooks contain two different '
        'reference-limit dictionaries. They are shown separately here so the dashboard does not '
        'silently combine incompatible assumptions. These are reference ranges used by the project, '
        'not a claim that every value represents a current official WHO drinking-water standard.</div>',
        unsafe_allow_html=True
    )

    profile_name = st.selectbox("Reference profile", list(WHO_PROFILES.keys()))
    limits = WHO_PROFILES[profile_name]

    violation_counts = {}
    violation_rates = {}

    for feature, (low, high) in limits.items():
        if feature not in df.columns:
            continue
        mask = (df[feature] < low) | (df[feature] > high)
        violation_counts[feature] = int(mask.sum())
        violation_rates[feature] = float(mask.mean() * 100)

    vdf = pd.DataFrame({
        "Feature": list(violation_counts.keys()),
        "Violations": list(violation_counts.values()),
        "Violation %": [violation_rates[x] for x in violation_counts]
    }).sort_values("Violations", ascending=False)

    c1, c2 = st.columns(2)
    with c1:
        metric_card("Total Feature Checks", f"{len(df) * len(vdf):,}")
    with c2:
        metric_card("Total Violations", f"{vdf['Violations'].sum():,}")

    fig = px.bar(
        vdf,
        x="Feature", y="Violation %",
        color="Violation %",
        title=f"Reference Range Violations — {profile_name}"
    )
    fig.update_layout(template="plotly_dark", xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(vdf, use_container_width=True, hide_index=True)

    st.markdown("### Reference Limits")
    limits_df = pd.DataFrame([
        {"Feature": k, "Minimum": v[0], "Maximum": v[1]}
        for k, v in limits.items()
    ])
    st.dataframe(limits_df, use_container_width=True, hide_index=True)

    # Per-row violation count
    violation_matrix = pd.DataFrame(index=df.index)
    for feature, (low, high) in limits.items():
        if feature in df.columns:
            violation_matrix[feature] = ((df[feature] < low) | (df[feature] > high)).astype(int)

    row_violation_count = violation_matrix.sum(axis=1)
    fig = px.histogram(
        row_violation_count,
        nbins=max(10, int(row_violation_count.max()) + 1),
        title="Number of Reference-Range Violations per Water Sample",
        labels={"value": "Violations per Sample"}
    )
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# 3D ANALYTICS
# ============================================================
elif page == "8️⃣ 3D Analytics":
    st.markdown("## 8. Interactive 3D Water Quality Analytics")

    st.write(
        "Use the controls to explore water samples in three dimensions. "
        "The first view uses original features; the second compresses the nine "
        "numeric quality variables into three PCA components."
    )

    st.markdown("### 8.1 Raw Feature 3D Scatter")
    a, b, c = st.columns(3)
    with a: x_feature = st.selectbox("X axis", FEATURES, index=0)
    with b: y_feature = st.selectbox("Y axis", FEATURES, index=1)
    with c: z_feature = st.selectbox("Z axis", FEATURES, index=2)

    sample_size = st.slider("Maximum points to display", 500, min(5000, len(df)), min(2500, len(df)))

    plot_df = df.sample(sample_size, random_state=42) if len(df) > sample_size else df.copy()
    plot_df["Potability Label"] = plot_df[TARGET].map({0: "Unsafe", 1: "Safe"})

    fig = px.scatter_3d(
        plot_df,
        x=x_feature, y=y_feature, z=z_feature,
        color="Potability Label",
        hover_data=FEATURES,
        title=f"3D Feature Space: {x_feature} × {y_feature} × {z_feature}"
    )
    fig.update_layout(template="plotly_dark", height=700)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 8.2 PCA 3D Projection")
    X = df[FEATURES].copy()
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=3, random_state=42)
    components = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame(
        components,
        columns=["PC1", "PC2", "PC3"]
    )
    pca_df["Potability"] = df[TARGET].map({0: "Unsafe", 1: "Safe"})

    explained = pca.explained_variance_ratio_ * 100

    fig = px.scatter_3d(
        pca_df,
        x="PC1", y="PC2", z="PC3",
        color="Potability",
        title="3D PCA Projection of Water Quality Samples"
    )
    fig.update_layout(template="plotly_dark", height=700)
    st.plotly_chart(fig, use_container_width=True)

    e1, e2, e3 = st.columns(3)
    with e1: metric_card("PC1 Variance", f"{explained[0]:.1f}%")
    with e2: metric_card("PC2 Variance", f"{explained[1]:.1f}%")
    with e3: metric_card("PC3 Variance", f"{explained[2]:.1f}%")

    st.markdown("### PCA Interpretation")
    st.write(
        "PCA is an additional visualization technique for this dashboard. "
        "It is useful for seeing whether safe and unsafe samples occupy visibly "
        "different regions of the standardized feature space. It does not itself "
        "prove causality or replace the project's original correlation analysis."
    )

# ============================================================
# INSIGHTS
# ============================================================
elif page == "9️⃣ Insights & Conclusion":
    st.markdown("## 9. Insights & Conclusion")

    corr = df.corr(numeric_only=True)[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)
    top_feature = corr.index[0]
    top_corr = corr.iloc[0]

    outlier_counts = {
        col: iqr_info(df, col)[0] for col in FEATURES
    }
    top_outlier = max(outlier_counts, key=outlier_counts.get)

    st.markdown("### Key Findings from the Supplied Project")
    findings = [
        "The dataset contains 3,276 water samples and 10 columns, including Potability as the target.",
        "The target class represents safe (1) and unsafe (0) water.",
        "Missing values are handled through numeric median imputation in the analysis-ready dataset.",
        f"The strongest absolute linear correlation with Potability in the loaded data is {top_feature} ({top_corr:.3f}).",
        f"The IQR method identifies {top_outlier} as the feature with the largest number of outliers in the current cleaned data.",
        "The project presentation emphasizes that potability is influenced by multiple weak signals rather than a single simple threshold.",
        "Reference-range analysis is included as a separate diagnostic layer rather than treating it as a predictive model.",
    ]

    for item in findings:
        st.markdown(f'<div class="insight">🔹 {item}</div>', unsafe_allow_html=True)

    st.markdown("### Project Conclusion")
    st.write(
        "Water quality should be evaluated across multiple chemical and physical parameters. "
        "The exploratory analysis helps identify distributions, class imbalance, correlations, "
        "outliers and reference-range violations. These findings provide a strong foundation "
        "for a future machine-learning model and real-time water-quality monitoring system."
    )

    st.markdown("### Future Scope")
    st.write(
        "The supplied presentation identifies machine learning and real-time monitoring as "
        "future extensions. A next stage could add a trained classification model, probability "
        "scores, model explainability, sensor/API ingestion and alerting."
    )

# ============================================================
# DATA EXPLORER
# ============================================================
elif page == "📋 Interactive Data Explorer":
    st.markdown("## 📋 Interactive Data Explorer")

    filtered = df.copy()

    col1, col2 = st.columns(2)
    with col1:
        potability_filter = st.multiselect(
            "Potability",
            [0, 1],
            default=[0, 1],
            format_func=lambda x: "Unsafe (0)" if x == 0 else "Safe (1)"
        )
    with col2:
        selected_cols = st.multiselect(
            "Columns",
            df.columns.tolist(),
            default=df.columns.tolist()
        )

    filtered = filtered[filtered[TARGET].isin(potability_filter)]

    st.write(f"Showing **{len(filtered):,}** rows.")
    st.dataframe(filtered[selected_cols], use_container_width=True, height=600)

    st.download_button(
        "⬇️ Download Filtered Data",
        filtered.to_csv(index=False).encode("utf-8"),
        "water_quality_filtered.csv",
        "text/csv"
    )

st.sidebar.divider()
st.sidebar.success("Dashboard ready")
