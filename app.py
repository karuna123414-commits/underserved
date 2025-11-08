import json
from typing import List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    recall_score,
    roc_auc_score,
    roc_curve,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---- Optional libraries (app will continue if not installed) ----
try:
    import shap  # type: ignore
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

try:
    from imblearn.over_sampling import SMOTE  # type: ignore
    SMOTE_AVAILABLE = True
except Exception:
    SMOTE_AVAILABLE = False

try:
    from fairlearn.metrics import (  # type: ignore
        demographic_parity_difference,
        equalized_odds_difference,
    )
    FAIRLEARN_AVAILABLE = True
except Exception:
    FAIRLEARN_AVAILABLE = False


# -------------------------
# Streamlit Page Config
# -------------------------
st.set_page_config(
    page_title="Underserved Communities — Interactive Dashboard",
    layout="wide",
    page_icon="🩺",
)


# -------------------------
# Utilities
# -------------------------
NUMERIC_SAMPLE_COLS = [
    "population",
    "median_income",
    "uninsured_rate",
    "education_below_highschool",
    "transportation_no_vehicle",
    "primary_care_physicians",
    "hospitals",
    "community_health_centers",
    "mental_health_providers",
    "diabetes_prevalence",
    "obesity_prevalence",
    "heart_disease_prevalence",
    "smoking_rate",
]


def make_synthetic(n_counties: int = 200, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic county-level dataset with an 'underserved' label."""
    rng = np.random.default_rng(seed)
    county_fips = [f"{i:05d}" for i in range(1001, 1001 + n_counties)]

    acs = pd.DataFrame(
        {
            "county_fips": county_fips,
            "population": rng.integers(10_000, 500_000, n_counties),
            "median_income": rng.integers(30_000, 90_000, n_counties),
            "uninsured_rate": np.round(rng.uniform(0.05, 0.25, n_counties), 3),
            "education_below_highschool": np.round(
                rng.uniform(0.10, 0.40, n_counties), 3
            ),
            "transportation_no_vehicle": np.round(
                rng.uniform(0.02, 0.15, n_counties), 3
            ),
        }
    )

    hrsa = pd.DataFrame(
        {
            "county_fips": county_fips,
            "primary_care_physicians": rng.integers(5, 200, n_counties),
            "hospitals": rng.integers(0, 10, n_counties),
            "community_health_centers": rng.integers(0, 5, n_counties),
            "mental_health_providers": rng.integers(1, 50, n_counties),
        }
    )

    brfss = pd.DataFrame(
        {
            "county_fips": county_fips,
            "diabetes_prevalence": np.round(
                rng.uniform(0.05, 0.20, n_counties), 3
            ),
            "obesity_prevalence": np.round(rng.uniform(0.20, 0.45, n_counties), 3),
            "heart_disease_prevalence": np.round(
                rng.uniform(0.03, 0.15, n_counties), 3
            ),
            "smoking_rate": np.round(rng.uniform(0.10, 0.30, n_counties), 3),
        }
    )

    df = acs.merge(hrsa, on="county_fips").merge(brfss, on="county_fips")

    # Create target
    df["underserved"] = np.where(
        (df["uninsured_rate"] > 0.15)
        & (df["primary_care_physicians"] < 50)
        & (df["diabetes_prevalence"] > 0.12),
        1,
        0,
    )

    # Synthetic coords for quick map demo
    df["latitude"] = np.round(rng.uniform(25, 49, df.shape[0]), 4)
    df["longitude"] = np.round(rng.uniform(-124, -67, df.shape[0]), 4)
    return df


def clean_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    return df.fillna(df.median(numeric_only=True))


def normalize_features(df: pd.DataFrame, columns: List[str]) -> Tuple[pd.DataFrame, StandardScaler]:
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[columns] = scaler.fit_transform(df_scaled[columns])
    return df_scaled, scaler


def train_rf(
    X_tr,
    y_tr,
    n_estimators: int = 300,
    max_depth=None,
    class_weight=None,
    random_state: int = 42,
):
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_tr, y_tr)
    return model


def make_confusion_heatmap(cm: np.ndarray, labels=("Served (0)", "Underserved (1)")) -> go.Figure:
    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=[f"Pred {labels[0]}", f"Pred {labels[1]}"],
            y=[f"True {labels[0]}", f"True {labels[1]}"],
            text=cm,
            texttemplate="%{text}",
            showscale=False,
        )
    )
    fig.update_layout(title="Confusion Matrix")
    return fig


def threshold_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    y_hat = (y_prob >= threshold).astype(int)
    acc = accuracy_score(y_true, y_hat)
    pr, rc, f1, _ = precision_recall_fscore_support(
        y_true, y_hat, average="binary", zero_division=0, pos_label=1
    )
    cm = confusion_matrix(y_true, y_hat)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = np.nan
    return {"accuracy": acc, "precision": pr, "recall": rc, "f1": f1, "cm": cm, "auc": auc, "y_hat": y_hat}


def fairness_by_group_simple(
    X_test: pd.DataFrame, y_true: np.ndarray, y_hat: np.ndarray, sensitive_col: str
) -> dict:
    """Median split fairness: demographic parity (selection-rate diff) and equal opportunity (recall)."""
    med = X_test[sensitive_col].median()
    group_A = X_test[sensitive_col] <= med  # disadvantaged
    group_B = X_test[sensitive_col] > med   # advantaged

    sel_A = float(np.mean(y_hat[group_A] == 1))
    sel_B = float(np.mean(y_hat[group_B] == 1))
    sel_diff = sel_A - sel_B

    def safe_recall(y_t, y_h):
        if np.sum(y_t == 1) == 0:
            return np.nan
        return recall_score(y_t, y_h, pos_label=1)

    rec_A = safe_recall(y_true[group_A], y_hat[group_A])
    rec_B = safe_recall(y_true[group_B], y_hat[group_B])

    return {
        "group_A_size": int(group_A.sum()),
        "group_B_size": int(group_B.sum()),
        "selection_rate_A": sel_A,
        "selection_rate_B": sel_B,
        "demographic_parity_diff": sel_diff,
        "equal_opportunity_recall_A": None if np.isnan(rec_A) else float(rec_A),
        "equal_opportunity_recall_B": None if np.isnan(rec_B) else float(rec_B),
    }


def render_html_report(summary: dict) -> bytes:
    """Create a simple HTML report and return it as bytes."""
    html = f"""
    <html><head><meta charset='utf-8'><title>Underserved Dashboard Report</title></head>
    <body>
    <h1>Underserved Dashboard — Summary Report</h1>
    <h2>Run Settings</h2>
    <ul>
      <li>Test Size: {summary.get('test_size')}</li>
      <li>Class Weight: {summary.get('class_weight')}</li>
      <li>Threshold: {summary.get('threshold')}</li>
      <li>n_estimators: {summary.get('n_estimators')}</li>
      <li>max_depth: {summary.get('max_depth')}</li>
      <li>SMOTE: {summary.get('smote_used')}</li>
    </ul>
    <h2>Classification Metrics</h2>
    <ul>
      <li>Accuracy: {summary.get('accuracy'):.3f}</li>
      <li>Precision (positive=underserved): {summary.get('precision'):.3f}</li>
      <li>Recall (positive=underserved): {summary.get('recall'):.3f}</li>
      <li>F1: {summary.get('f1'):.3f}</li>
      <li>AUC: {summary.get('auc')}</li>
    </ul>
    <h2>Fairness</h2>
    <pre>{summary.get('fairness_json')}</pre>
    <p><em>This auto-generated report is designed for quick inclusion in your milestone write-up. Add context and screenshots from the app for full credit.</em></p>
    </body></html>
    """
    return html.encode("utf-8")


# -------------------------
# Sidebar Controls
# -------------------------
st.sidebar.header("⚙️ Controls")

data_source = st.sidebar.radio(
    "Data Source",
    ["Upload CSV", "Use Synthetic Demo"],
    help="Upload your merged county-level CSV (include 'underserved' target) or use a synthetic example.",
)

if data_source == "Upload CSV":
    f = st.sidebar.file_uploader("Upload CSV (must contain target 'underserved')", type=["csv"])
    df_raw = pd.read_csv(f) if f is not None else None
else:
    n = st.sidebar.slider("Synthetic sample size (counties)", 100, 2000, 200, step=50)
    df_raw = make_synthetic(n_counties=n)

st.sidebar.markdown("---")
test_size = st.sidebar.slider("Test size (%)", 10, 40, 20, step=5) / 100.0
threshold = st.sidebar.slider("Decision Threshold (for underserved=1)", 0.0, 1.0, 0.50, 0.01)

class_weight_opt = st.sidebar.selectbox(
    "Class weighting",
    ["None", "balanced"],
    help="Try 'balanced' when underserved is rare.",
)
class_weight = None if class_weight_opt == "None" else "balanced"

smote_enabled = st.sidebar.checkbox(
    "Use SMOTE on training set (if minority is rare)", value=False, help="Requires imbalanced-learn"
)

st.sidebar.markdown("### Random Forest")
n_estimators = st.sidebar.slider("n_estimators", 100, 800, 300, step=50)
max_depth = st.sidebar.select_slider("max_depth", options=[None, 5, 10, 15, 20, 30], value=None)

st.sidebar.markdown("---")
k_clusters = st.sidebar.slider("KMeans: number of clusters (k)", 2, 8, 3, step=1)


# -------------------------
# Main Layout
# -------------------------
st.title("🩺 Identifying Underserved Communities — Interactive Dashboard")
st.caption("Predict and explore underserved communities with interpretable, equity-aware metrics.")

if df_raw is None:
    st.info("Upload a CSV to begin. Or switch to **Use Synthetic Demo** in the sidebar for a pre-filled example.")
    st.stop()

# Validate minimum columns
required_cols = set(NUMERIC_SAMPLE_COLS + ["underserved"])
missing_cols = required_cols - set(df_raw.columns)
if missing_cols:
    st.warning(
        f"The dataset is missing some expected columns: {sorted(list(missing_cols))}. "
        "You can still proceed if you have similar columns—select them below."
    )
    numeric_cols = [c for c in df_raw.columns if pd.api.types.is_numeric_dtype(df_raw[c])]
    default_feats = [c for c in NUMERIC_SAMPLE_COLS if c in numeric_cols]
else:
    numeric_cols = [c for c in df_raw.columns if pd.api.types.is_numeric_dtype(df_raw[c])]
    default_feats = [c for c in NUMERIC_SAMPLE_COLS if c in df_raw.columns]

with st.expander("🔧 Feature & Target Selection", expanded=True):
    feat_cols = st.multiselect(
        "Choose feature columns",
        options=numeric_cols,
        default=default_feats if default_feats else numeric_cols[: min(12, len(numeric_cols))],
        help="Pick your model input features. Numeric columns only.",
    )
    target_col = st.selectbox(
        "Target column (binary: 0 = served, 1 = underserved)",
        options=[c for c in df_raw.columns if c in ["underserved"] or pd.api.types.is_bool_dtype(df_raw[c])],
        index=0 if "underserved" in df_raw.columns else 0,
    )
    lat_col = st.selectbox("Latitude column (optional)", options=["(none)"] + list(df_raw.columns), index=0)
    lon_col = st.selectbox("Longitude column (optional)", options=["(none)"] + list(df_raw.columns), index=0)
    sensitive_col = st.selectbox(
        "Sensitive (proxy) feature for fairness check",
        options=feat_cols if "median_income" not in df_raw.columns else ["median_income"] + feat_cols,
        index=0,
    )

# Data preview
st.subheader("📄 Data Preview")
st.dataframe(df_raw.head(20), use_container_width=True)

# Preprocess
df = df_raw.copy()
df = clean_missing_values(df)

if not feat_cols:
    st.error("Please select at least one feature column to continue.")
    st.stop()

X = df[feat_cols].copy()
y = df[target_col].astype(int).values

# Normalize (fit on train only)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42, stratify=y if len(np.unique(y)) == 2 else None
)
X_train_scaled, scaler = normalize_features(X_train, feat_cols)
X_test_scaled = X_test.copy()
X_test_scaled[feat_cols] = scaler.transform(X_test)

# Optional SMOTE
smote_used = False
if smote_enabled:
    if SMOTE_AVAILABLE:
        try:
            sm = SMOTE(random_state=42)
            X_res, y_res = sm.fit_resample(X_train_scaled, y_train)
            X_tr_final, y_tr_final = X_res, y_res
            smote_used = True
        except Exception as e:
            st.warning(f"SMOTE failed: {e}. Training without SMOTE.")
            X_tr_final, y_tr_final = X_train_scaled, y_train
    else:
        st.info("imblearn not installed; cannot apply SMOTE. Install 'imbalanced-learn' to enable.")
        X_tr_final, y_tr_final = X_train_scaled, y_train
else:
    X_tr_final, y_tr_final = X_train_scaled, y_train

# Train model
rf = train_rf(
    X_tr_final,
    y_tr_final,
    n_estimators=n_estimators,
    max_depth=max_depth,
    class_weight=class_weight,
)

# Predict probs on test
if hasattr(rf, "predict_proba"):
    y_prob = rf.predict_proba(X_test_scaled)[:, 1]
else:
    try:
        y_prob = rf.decision_function(X_test_scaled)
    except Exception:
        y_prob = rf.predict(X_test_scaled)

metrics = threshold_metrics(y_test, y_prob, threshold=threshold)

# Tabs
tab_overview, tab_model, tab_thresholds, tab_fairness, tab_clusters, tab_map, tab_export = st.tabs(
    ["Overview", "Modeling", "Threshold & ROC", "Fairness", "Clustering", "Map", "Export"]
)

with tab_overview:
    st.markdown("### ✨ Summary")
    left, right = st.columns([1, 1])
    with left:
        st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
        st.metric("Precision (Underserved=1)", f"{metrics['precision']:.3f}")
    with right:
        st.metric("Recall (Underserved=1)", f"{metrics['recall']:.3f}")
        st.metric("F1-score", f"{metrics['f1']:.3f}")

    # Feature importance
    try:
        importances = rf.feature_importances_
        imp_df = pd.DataFrame({"feature": feat_cols, "importance": importances}).sort_values(
            "importance", ascending=False
        )
        st.markdown("#### Feature Importance")
        st.bar_chart(imp_df.set_index("feature"))
    except Exception:
        st.info("Feature importances unavailable for this model.")

    st.markdown(
        "This dashboard prioritizes **equity**. If underserved cases are rare, overall accuracy can be misleading. "
        "Use **Threshold & ROC** to raise recall for the underserved class, and verify **Fairness** across income (or other proxy) groups."
    )

with tab_model:
    st.markdown("### Classification Report (at current threshold)")
    y_hat = metrics["y_hat"]
    report = classification_report(y_test, y_hat, zero_division=0, output_dict=True)
    rep_df = pd.DataFrame(report).T
    st.dataframe(rep_df.style.format(precision=3), use_container_width=True)

    fig_cm = make_confusion_heatmap(metrics["cm"])
    st.plotly_chart(fig_cm, use_container_width=True)

    # SHAP explainability
    st.markdown("### Explainability (SHAP)")
    if SHAP_AVAILABLE:
        try:
            explainer = shap.Explainer(rf, X_tr_final, feature_names=feat_cols)
            shap_values = explainer(X_test_scaled, check_additivity=False)

            st.write("**Summary (bar)** — average absolute impact per feature:")
            fig_bar = shap.plots.bar(shap_values, show=False)
            st.pyplot(fig_bar, use_container_width=True)

            st.write("**Single County Force Plot** — how features push prediction for one example:")
            row_idx = 0
            exp_row = explainer(X_test_scaled.iloc[[row_idx]], check_additivity=False)
            try:
                fig_force = shap.plots.force(
                    exp_row[0].base_values,
                    exp_row[0].values,
                    X_test_scaled.iloc[[row_idx]],
                    matplotlib=True,
                    show=False,
                )
            except Exception:
                fig_force = shap.plots.force(
                    exp_row.base_values[0],
                    exp_row.values[0],
                    X_test_scaled.iloc[[row_idx]],
                    matplotlib=True,
                    show=False,
                )
            st.pyplot(fig_force, use_container_width=True)
        except Exception as e:
            st.warning(f"SHAP explanation failed: {e}")
    else:
        st.info("SHAP not installed. Add `shap` to requirements to enable.")

with tab_thresholds:
    st.markdown("### Decision Threshold Tuning")
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_fig = go.Figure()
    roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC"))
    roc_fig.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance", line=dict(dash="dash"))
    )
    auc_display = "N/A" if np.isnan(metrics["auc"]) else f"{metrics['auc']:.3f}"
    roc_fig.update_layout(
        title=f"ROC Curve — AUC: {auc_display}",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
    )
    st.plotly_chart(roc_fig, use_container_width=True)

    # Threshold sweep
    thresholds = np.arange(0.0, 1.01, 0.05)
    rows = []
    for t in thresholds:
        m = threshold_metrics(y_test, y_prob, threshold=float(t))
        rows.append([t, m["accuracy"], m["precision"], m["recall"], m["f1"]])
    tdf = pd.DataFrame(rows, columns=["threshold", "accuracy", "precision", "recall", "f1"]).round(3)
    st.dataframe(tdf, use_container_width=True)
    st.caption(
        "Tip: For equity, aim for higher **recall** on the underserved class while keeping precision reasonable."
    )

with tab_fairness:
    st.markdown("### Fairness / Equity Check")
    if sensitive_col not in X_test_scaled.columns:
        st.info("Pick a valid sensitive (proxy) column in the settings above.")
    else:
        fair_simple = fairness_by_group_simple(X_test_scaled, y_test, metrics["y_hat"], sensitive_col)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Group A size (≤ median)", fair_simple["group_A_size"])
        c2.metric("Group B size (> median)", fair_simple["group_B_size"])
        c3.metric("Demographic parity (A−B)", f"{fair_simple['demographic_parity_diff']:.3f}")
        if (fair_simple["equal_opportunity_recall_A"] is None) or (
            fair_simple["equal_opportunity_recall_B"] is None
        ):
            c4.metric("Recall gap (A vs B)", "N/A")
        else:
            gap = fair_simple["equal_opportunity_recall_A"] - fair_simple["equal_opportunity_recall_B"]
            c4.metric("Recall gap (A vs B)", f"{gap:.3f}")
        st.json(fair_simple)
        st.caption("Group A = counties at or below the median of the chosen sensitive feature (e.g., lower-income).")

        st.markdown("#### Fairlearn metrics")
        if FAIRLEARN_AVAILABLE:
            sens_bin = (X_test_scaled[sensitive_col] <= X_test_scaled[sensitive_col].median()).astype(int)
            dp_diff = demographic_parity_difference(
                y_true=y_test, y_pred=metrics["y_hat"], sensitive_features=sens_bin
            )
            try:
                eo_diff = equalized_odds_difference(
                    y_true=y_test, y_pred=metrics["y_hat"], sensitive_features=sens_bin
                )
            except Exception:
                eo_diff = None
            st.write(
                {
                    "fairlearn.demographic_parity_difference": float(dp_diff),
                    "fairlearn.equalized_odds_difference": None if eo_diff is None else float(eo_diff),
                }
            )
        else:
            st.info("Install `fairlearn` to compute additional fairness metrics.")

with tab_clusters:
    st.markdown("### Clustering (K-Means)")
    scaler_all = StandardScaler().fit(X[feat_cols])
    X_all_scaled = scaler_all.transform(X[feat_cols])

    # Compatibility for different sklearn versions
    try:
        kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init="auto")
    except TypeError:
        kmeans = KMeans(n_clusters=k_clusters, random_state=42)

    labels = kmeans.fit_predict(X_all_scaled)
    sil = silhouette_score(X_all_scaled, labels)
    st.metric(f"Silhouette Score (k={k_clusters})", f"{sil:.3f}")

    pca = PCA(n_components=2, random_state=42)
    X2 = pca.fit_transform(X_all_scaled)
    df_plot = pd.DataFrame(
        {
            "pc1": X2[:, 0],
            "pc2": X2[:, 1],
            "cluster": labels.astype(str),
            "underserved": df[target_col].astype(int),
        }
    )
    fig = px.scatter(
        df_plot,
        x="pc1",
        y="pc2",
        color="cluster",
        symbol="underserved",
        title="PCA of Features colored by Cluster (symbols = underserved)",
    )
    st.plotly_chart(fig, use_container_width=True)

with tab_map:
    st.markdown("### Map")
    if (
        lat_col != "(none)"
        and lon_col != "(none)"
        and lat_col in df.columns
        and lon_col in df.columns
    ):
        map_df = df[[lat_col, lon_col, target_col]].copy()
        map_df.rename(columns={lat_col: "lat", lon_col: "lon"}, inplace=True)
        st.map(
            map_df.rename(columns={target_col: "is_underserved"}),
            latitude="lat",
            longitude="lon",
            size=20,
            color=None,
        )
    else:
        st.info("Provide latitude/longitude columns (or use synthetic demo) to view a map.")

with tab_export:
    st.markdown("### Export Predictions & Report")

    # Predictions for ALL rows
    full_scaled = scaler.transform(X[feat_cols])
    full_prob = (
        rf.predict_proba(full_scaled)[:, 1] if hasattr(rf, "predict_proba") else rf.predict(full_scaled)
    )
    full_pred = (full_prob >= threshold).astype(int)

    out = df.copy()
    out["underserved_pred_prob"] = full_prob
    out["underserved_pred"] = full_pred
    csv_bytes = out.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download CSV with Predictions",
        data=csv_bytes,
        file_name="underserved_predictions.csv",
        mime="text/csv",
    )

    # HTML report
    fair_json = json.dumps(fair_simple if "fair_simple" in locals() else {}, indent=2)
    summary = {
        "test_size": test_size,
        "class_weight": class_weight,
        "threshold": threshold,
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "smote_used": smote_used,
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "auc": None if np.isnan(metrics["auc"]) else float(metrics["auc"]),
        "fairness_json": fair_json,
    }
    html_bytes = render_html_report(summary)
    st.download_button(
        "📝 Download HTML Report",
        data=html_bytes,
        file_name="underserved_report.html",
        mime="text/html",
    )

st.markdown("---")
st.caption(
    "© 2025 — Capstone Dashboard. This template emphasizes transparency, fairness checks, threshold tuning, and exportable artifacts."
)
import json
from typing import List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    recall_score,
    roc_auc_score,
    roc_curve,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---- Optional libraries (app will continue if not installed) ----
try:
    import shap  # type: ignore
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

try:
    from imblearn.over_sampling import SMOTE  # type: ignore
    SMOTE_AVAILABLE = True
except Exception:
    SMOTE_AVAILABLE = False

try:
    from fairlearn.metrics import (  # type: ignore
        demographic_parity_difference,
        equalized_odds_difference,
    )
    FAIRLEARN_AVAILABLE = True
except Exception:
    FAIRLEARN_AVAILABLE = False


# -------------------------
# Streamlit Page Config
# -------------------------
st.set_page_config(
    page_title="Underserved Communities — Interactive Dashboard",
    layout="wide",
    page_icon="🩺",
)


# -------------------------
# Utilities
# -------------------------
NUMERIC_SAMPLE_COLS = [
    "population",
    "median_income",
    "uninsured_rate",
    "education_below_highschool",
    "transportation_no_vehicle",
    "primary_care_physicians",
    "hospitals",
    "community_health_centers",
    "mental_health_providers",
    "diabetes_prevalence",
    "obesity_prevalence",
    "heart_disease_prevalence",
    "smoking_rate",
]


def make_synthetic(n_counties: int = 200, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic county-level dataset with an 'underserved' label."""
    rng = np.random.default_rng(seed)
    county_fips = [f"{i:05d}" for i in range(1001, 1001 + n_counties)]

    acs = pd.DataFrame(
        {
            "county_fips": county_fips,
            "population": rng.integers(10_000, 500_000, n_counties),
            "median_income": rng.integers(30_000, 90_000, n_counties),
            "uninsured_rate": np.round(rng.uniform(0.05, 0.25, n_counties), 3),
            "education_below_highschool": np.round(
                rng.uniform(0.10, 0.40, n_counties), 3
            ),
            "transportation_no_vehicle": np.round(
                rng.uniform(0.02, 0.15, n_counties), 3
            ),
        }
    )

    hrsa = pd.DataFrame(
        {
            "county_fips": county_fips,
            "primary_care_physicians": rng.integers(5, 200, n_counties),
            "hospitals": rng.integers(0, 10, n_counties),
            "community_health_centers": rng.integers(0, 5, n_counties),
            "mental_health_providers": rng.integers(1, 50, n_counties),
        }
    )

    brfss = pd.DataFrame(
        {
            "county_fips": county_fips,
            "diabetes_prevalence": np.round(
                rng.uniform(0.05, 0.20, n_counties), 3
            ),
            "obesity_prevalence": np.round(rng.uniform(0.20, 0.45, n_counties), 3),
            "heart_disease_prevalence": np.round(
                rng.uniform(0.03, 0.15, n_counties), 3
            ),
            "smoking_rate": np.round(rng.uniform(0.10, 0.30, n_counties), 3),
        }
    )

    df = acs.merge(hrsa, on="county_fips").merge(brfss, on="county_fips")

    # Create target
    df["underserved"] = np.where(
        (df["uninsured_rate"] > 0.15)
        & (df["primary_care_physicians"] < 50)
        & (df["diabetes_prevalence"] > 0.12),
        1,
        0,
    )

    # Synthetic coords for quick map demo
    df["latitude"] = np.round(rng.uniform(25, 49, df.shape[0]), 4)
    df["longitude"] = np.round(rng.uniform(-124, -67, df.shape[0]), 4)
    return df


def clean_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    return df.fillna(df.median(numeric_only=True))


def normalize_features(df: pd.DataFrame, columns: List[str]) -> Tuple[pd.DataFrame, StandardScaler]:
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[columns] = scaler.fit_transform(df_scaled[columns])
    return df_scaled, scaler


def train_rf(
    X_tr,
    y_tr,
    n_estimators: int = 300,
    max_depth=None,
    class_weight=None,
    random_state: int = 42,
):
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_tr, y_tr)
    return model


def make_confusion_heatmap(cm: np.ndarray, labels=("Served (0)", "Underserved (1)")) -> go.Figure:
    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=[f"Pred {labels[0]}", f"Pred {labels[1]}"],
            y=[f"True {labels[0]}", f"True {labels[1]}"],
            text=cm,
            texttemplate="%{text}",
            showscale=False,
        )
    )
    fig.update_layout(title="Confusion Matrix")
    return fig


def threshold_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    y_hat = (y_prob >= threshold).astype(int)
    acc = accuracy_score(y_true, y_hat)
    pr, rc, f1, _ = precision_recall_fscore_support(
        y_true, y_hat, average="binary", zero_division=0, pos_label=1
    )
    cm = confusion_matrix(y_true, y_hat)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = np.nan
    return {"accuracy": acc, "precision": pr, "recall": rc, "f1": f1, "cm": cm, "auc": auc, "y_hat": y_hat}


def fairness_by_group_simple(
    X_test: pd.DataFrame, y_true: np.ndarray, y_hat: np.ndarray, sensitive_col: str
) -> dict:
    """Median split fairness: demographic parity (selection-rate diff) and equal opportunity (recall)."""
    med = X_test[sensitive_col].median()
    group_A = X_test[sensitive_col] <= med  # disadvantaged
    group_B = X_test[sensitive_col] > med   # advantaged

    sel_A = float(np.mean(y_hat[group_A] == 1))
    sel_B = float(np.mean(y_hat[group_B] == 1))
    sel_diff = sel_A - sel_B

    def safe_recall(y_t, y_h):
        if np.sum(y_t == 1) == 0:
            return np.nan
        return recall_score(y_t, y_h, pos_label=1)

    rec_A = safe_recall(y_true[group_A], y_hat[group_A])
    rec_B = safe_recall(y_true[group_B], y_hat[group_B])

    return {
        "group_A_size": int(group_A.sum()),
        "group_B_size": int(group_B.sum()),
        "selection_rate_A": sel_A,
        "selection_rate_B": sel_B,
        "demographic_parity_diff": sel_diff,
        "equal_opportunity_recall_A": None if np.isnan(rec_A) else float(rec_A),
        "equal_opportunity_recall_B": None if np.isnan(rec_B) else float(rec_B),
    }


def render_html_report(summary: dict) -> bytes:
    """Create a simple HTML report and return it as bytes."""
    html = f"""
    <html><head><meta charset='utf-8'><title>Underserved Dashboard Report</title></head>
    <body>
    <h1>Underserved Dashboard — Summary Report</h1>
    <h2>Run Settings</h2>
    <ul>
      <li>Test Size: {summary.get('test_size')}</li>
      <li>Class Weight: {summary.get('class_weight')}</li>
      <li>Threshold: {summary.get('threshold')}</li>
      <li>n_estimators: {summary.get('n_estimators')}</li>
      <li>max_depth: {summary.get('max_depth')}</li>
      <li>SMOTE: {summary.get('smote_used')}</li>
    </ul>
    <h2>Classification Metrics</h2>
    <ul>
      <li>Accuracy: {summary.get('accuracy'):.3f}</li>
      <li>Precision (positive=underserved): {summary.get('precision'):.3f}</li>
      <li>Recall (positive=underserved): {summary.get('recall'):.3f}</li>
      <li>F1: {summary.get('f1'):.3f}</li>
      <li>AUC: {summary.get('auc')}</li>
    </ul>
    <h2>Fairness</h2>
    <pre>{summary.get('fairness_json')}</pre>
    <p><em>This auto-generated report is designed for quick inclusion in your milestone write-up. Add context and screenshots from the app for full credit.</em></p>
    </body></html>
    """
    return html.encode("utf-8")


# -------------------------
# Sidebar Controls
# -------------------------
st.sidebar.header("⚙️ Controls")

data_source = st.sidebar.radio(
    "Data Source",
    ["Upload CSV", "Use Synthetic Demo"],
    help="Upload your merged county-level CSV (include 'underserved' target) or use a synthetic example.",
    key="sidebar_data_source"
)

if data_source == "Upload CSV":
    f = st.sidebar.file_uploader("Upload CSV (must contain target 'underserved')", type=["csv"], key="sidebar_csv_uploader")
    df_raw = pd.read_csv(f) if f is not None else None
else:
    n = st.sidebar.slider("Synthetic sample size (counties)", 100, 2000, 200, step=50)
    df_raw = make_synthetic(n_counties=n)

st.sidebar.markdown("---")
test_size = st.sidebar.slider("Test size (%)", 10, 40, 20, step=5, key="sidebar_slider_source") / 100.0
threshold = st.sidebar.slider("Decision Threshold (for underserved=1)", 0.0, 1.0, 0.50, 0.01, key="sidebar_slider_source2")

class_weight_opt = st.sidebar.selectbox(
    "Class weighting",
    ["None", "balanced"],
    help="Try 'balanced' when underserved is rare.",
    key="sidebar_selectbox_source"
)
class_weight = None if class_weight_opt == "None" else "balanced"

smote_enabled = st.sidebar.checkbox(
    "Use SMOTE on training set (if minority is rare)", value=False, help="Requires imbalanced-learn"
)

st.sidebar.markdown("### Random Forest")
n_estimators = st.sidebar.slider("n_estimators", 100, 800, 300, step=50)
max_depth = st.sidebar.select_slider("max_depth", options=[None, 5, 10, 15, 20, 30], value=None)

st.sidebar.markdown("---")
k_clusters = st.sidebar.slider("KMeans: number of clusters (k)", 2, 8, 3, step=1)


# -------------------------
# Main Layout
# -------------------------
st.title("🩺 Identifying Underserved Communities — Interactive Dashboard")
st.caption("Predict and explore underserved communities with interpretable, equity-aware metrics.")

if df_raw is None:
    st.info("Upload a CSV to begin. Or switch to **Use Synthetic Demo** in the sidebar for a pre-filled example.")
    st.stop()

# Validate minimum columns
required_cols = set(NUMERIC_SAMPLE_COLS + ["underserved"])
missing_cols = required_cols - set(df_raw.columns)
if missing_cols:
    st.warning(
        f"The dataset is missing some expected columns: {sorted(list(missing_cols))}. "
        "You can still proceed if you have similar columns—select them below."
    )
    numeric_cols = [c for c in df_raw.columns if pd.api.types.is_numeric_dtype(df_raw[c])]
    default_feats = [c for c in NUMERIC_SAMPLE_COLS if c in numeric_cols]
else:
    numeric_cols = [c for c in df_raw.columns if pd.api.types.is_numeric_dtype(df_raw[c])]
    default_feats = [c for c in NUMERIC_SAMPLE_COLS if c in df_raw.columns]

with st.expander("🔧 Feature & Target Selection", expanded=True):
    feat_cols = st.multiselect(
        "Choose feature columns",
        options=numeric_cols,
        default=default_feats if default_feats else numeric_cols[: min(12, len(numeric_cols))],
        help="Pick your model input features. Numeric columns only.",
    )
    target_col = st.selectbox(
        "Target column (binary: 0 = served, 1 = underserved)",
        options=[c for c in df_raw.columns if c in ["underserved"] or pd.api.types.is_bool_dtype(df_raw[c])],
        index=0 if "underserved" in df_raw.columns else 0,
    )
    lat_col = st.selectbox("Latitude column (optional)", options=["(none)"] + list(df_raw.columns), index=0)
    lon_col = st.selectbox("Longitude column (optional)", options=["(none)"] + list(df_raw.columns), index=0)
    sensitive_col = st.selectbox(
        "Sensitive (proxy) feature for fairness check",
        options=feat_cols if "median_income" not in df_raw.columns else ["median_income"] + feat_cols,
        index=0,
    )

# Data preview
st.subheader("📄 Data Preview")
st.dataframe(df_raw.head(20), use_container_width=True)

# Preprocess
df = df_raw.copy()
df = clean_missing_values(df)

if not feat_cols:
    st.error("Please select at least one feature column to continue.")
    st.stop()

X = df[feat_cols].copy()
y = df[target_col].astype(int).values

# Normalize (fit on train only)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42, stratify=y if len(np.unique(y)) == 2 else None
)
X_train_scaled, scaler = normalize_features(X_train, feat_cols)
X_test_scaled = X_test.copy()
X_test_scaled[feat_cols] = scaler.transform(X_test)

# Optional SMOTE
smote_used = False
if smote_enabled:
    if SMOTE_AVAILABLE:
        try:
            sm = SMOTE(random_state=42)
            X_res, y_res = sm.fit_resample(X_train_scaled, y_train)
            X_tr_final, y_tr_final = X_res, y_res
            smote_used = True
        except Exception as e:
            st.warning(f"SMOTE failed: {e}. Training without SMOTE.")
            X_tr_final, y_tr_final = X_train_scaled, y_train
    else:
        st.info("imblearn not installed; cannot apply SMOTE. Install 'imbalanced-learn' to enable.")
        X_tr_final, y_tr_final = X_train_scaled, y_train
else:
    X_tr_final, y_tr_final = X_train_scaled, y_train

# Train model
rf = train_rf(
    X_tr_final,
    y_tr_final,
    n_estimators=n_estimators,
    max_depth=max_depth,
    class_weight=class_weight,
)

# Predict probs on test
if hasattr(rf, "predict_proba"):
    y_prob = rf.predict_proba(X_test_scaled)[:, 1]
else:
    try:
        y_prob = rf.decision_function(X_test_scaled)
    except Exception:
        y_prob = rf.predict(X_test_scaled)

metrics = threshold_metrics(y_test, y_prob, threshold=threshold)

# Tabs
tab_overview, tab_model, tab_thresholds, tab_fairness, tab_clusters, tab_map, tab_export = st.tabs(
    ["Overview", "Modeling", "Threshold & ROC", "Fairness", "Clustering", "Map", "Export"]
)

with tab_overview:
    st.markdown("### ✨ Summary")
    left, right = st.columns([1, 1])
    with left:
        st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
        st.metric("Precision (Underserved=1)", f"{metrics['precision']:.3f}")
    with right:
        st.metric("Recall (Underserved=1)", f"{metrics['recall']:.3f}")
        st.metric("F1-score", f"{metrics['f1']:.3f}")

    # Feature importance
    try:
        importances = rf.feature_importances_
        imp_df = pd.DataFrame({"feature": feat_cols, "importance": importances}).sort_values(
            "importance", ascending=False
        )
        st.markdown("#### Feature Importance")
        st.bar_chart(imp_df.set_index("feature"))
    except Exception:
        st.info("Feature importances unavailable for this model.")

    st.markdown(
        "This dashboard prioritizes **equity**. If underserved cases are rare, overall accuracy can be misleading. "
        "Use **Threshold & ROC** to raise recall for the underserved class, and verify **Fairness** across income (or other proxy) groups."
    )

with tab_model:
    st.markdown("### Classification Report (at current threshold)")
    y_hat = metrics["y_hat"]
    report = classification_report(y_test, y_hat, zero_division=0, output_dict=True)
    rep_df = pd.DataFrame(report).T
    st.dataframe(rep_df.style.format(precision=3), use_container_width=True)

    fig_cm = make_confusion_heatmap(metrics["cm"])
    st.plotly_chart(fig_cm, use_container_width=True)

    # SHAP explainability
    st.markdown("### Explainability (SHAP)")
    if SHAP_AVAILABLE:
        try:
            explainer = shap.Explainer(rf, X_tr_final, feature_names=feat_cols)
            shap_values = explainer(X_test_scaled, check_additivity=False)

            st.write("**Summary (bar)** — average absolute impact per feature:")
            fig_bar = shap.plots.bar(shap_values, show=False)
            st.pyplot(fig_bar, use_container_width=True)

            st.write("**Single County Force Plot** — how features push prediction for one example:")
            row_idx = 0
            exp_row = explainer(X_test_scaled.iloc[[row_idx]], check_additivity=False)
            try:
                fig_force = shap.plots.force(
                    exp_row[0].base_values,
                    exp_row[0].values,
                    X_test_scaled.iloc[[row_idx]],
                    matplotlib=True,
                    show=False,
                )
            except Exception:
                fig_force = shap.plots.force(
                    exp_row.base_values[0],
                    exp_row.values[0],
                    X_test_scaled.iloc[[row_idx]],
                    matplotlib=True,
                    show=False,
                )
            st.pyplot(fig_force, use_container_width=True)
        except Exception as e:
            st.warning(f"SHAP explanation failed: {e}")
    else:
        st.info("SHAP not installed. Add `shap` to requirements to enable.")

with tab_thresholds:
    st.markdown("### Decision Threshold Tuning")
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_fig = go.Figure()
    roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC"))
    roc_fig.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance", line=dict(dash="dash"))
    )
    auc_display = "N/A" if np.isnan(metrics["auc"]) else f"{metrics['auc']:.3f}"
    roc_fig.update_layout(
        title=f"ROC Curve — AUC: {auc_display}",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
    )
    st.plotly_chart(roc_fig, use_container_width=True)

    # Threshold sweep
    thresholds = np.arange(0.0, 1.01, 0.05)
    rows = []
    for t in thresholds:
        m = threshold_metrics(y_test, y_prob, threshold=float(t))
        rows.append([t, m["accuracy"], m["precision"], m["recall"], m["f1"]])
    tdf = pd.DataFrame(rows, columns=["threshold", "accuracy", "precision", "recall", "f1"]).round(3)
    st.dataframe(tdf, use_container_width=True)
    st.caption(
        "Tip: For equity, aim for higher **recall** on the underserved class while keeping precision reasonable."
    )

with tab_fairness:
    st.markdown("### Fairness / Equity Check")
    if sensitive_col not in X_test_scaled.columns:
        st.info("Pick a valid sensitive (proxy) column in the settings above.")
    else:
        fair_simple = fairness_by_group_simple(X_test_scaled, y_test, metrics["y_hat"], sensitive_col)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Group A size (≤ median)", fair_simple["group_A_size"])
        c2.metric("Group B size (> median)", fair_simple["group_B_size"])
        c3.metric("Demographic parity (A−B)", f"{fair_simple['demographic_parity_diff']:.3f}")
        if (fair_simple["equal_opportunity_recall_A"] is None) or (
            fair_simple["equal_opportunity_recall_B"] is None
        ):
            c4.metric("Recall gap (A vs B)", "N/A")
        else:
            gap = fair_simple["equal_opportunity_recall_A"] - fair_simple["equal_opportunity_recall_B"]
            c4.metric("Recall gap (A vs B)", f"{gap:.3f}")
        st.json(fair_simple)
        st.caption("Group A = counties at or below the median of the chosen sensitive feature (e.g., lower-income).")

        st.markdown("#### Fairlearn metrics")
        if FAIRLEARN_AVAILABLE:
            sens_bin = (X_test_scaled[sensitive_col] <= X_test_scaled[sensitive_col].median()).astype(int)
            dp_diff = demographic_parity_difference(
                y_true=y_test, y_pred=metrics["y_hat"], sensitive_features=sens_bin
            )
            try:
                eo_diff = equalized_odds_difference(
                    y_true=y_test, y_pred=metrics["y_hat"], sensitive_features=sens_bin
                )
            except Exception:
                eo_diff = None
            st.write(
                {
                    "fairlearn.demographic_parity_difference": float(dp_diff),
                    "fairlearn.equalized_odds_difference": None if eo_diff is None else float(eo_diff),
                }
            )
        else:
            st.info("Install `fairlearn` to compute additional fairness metrics.")

with tab_clusters:
    st.markdown("### Clustering (K-Means)")
    scaler_all = StandardScaler().fit(X[feat_cols])
    X_all_scaled = scaler_all.transform(X[feat_cols])

    # Compatibility for different sklearn versions
    try:
        kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init="auto")
    except TypeError:
        kmeans = KMeans(n_clusters=k_clusters, random_state=42)

    labels = kmeans.fit_predict(X_all_scaled)
    sil = silhouette_score(X_all_scaled, labels)
    st.metric(f"Silhouette Score (k={k_clusters})", f"{sil:.3f}")

    pca = PCA(n_components=2, random_state=42)
    X2 = pca.fit_transform(X_all_scaled)
    df_plot = pd.DataFrame(
        {
            "pc1": X2[:, 0],
            "pc2": X2[:, 1],
            "cluster": labels.astype(str),
            "underserved": df[target_col].astype(int),
        }
    )
    fig = px.scatter(
        df_plot,
        x="pc1",
        y="pc2",
        color="cluster",
        symbol="underserved",
        title="PCA of Features colored by Cluster (symbols = underserved)",
    )
    st.plotly_chart(fig, use_container_width=True)

with tab_map:
    st.markdown("### Map")
    if (
        lat_col != "(none)"
        and lon_col != "(none)"
        and lat_col in df.columns
        and lon_col in df.columns
    ):
        map_df = df[[lat_col, lon_col, target_col]].copy()
        map_df.rename(columns={lat_col: "lat", lon_col: "lon"}, inplace=True)
        st.map(
            map_df.rename(columns={target_col: "is_underserved"}),
            latitude="lat",
            longitude="lon",
            size=20,
            color=None,
        )
    else:
        st.info("Provide latitude/longitude columns (or use synthetic demo) to view a map.")

with tab_export:
    st.markdown("### Export Predictions & Report")

    # Predictions for ALL rows
    full_scaled = scaler.transform(X[feat_cols])
    full_prob = (
        rf.predict_proba(full_scaled)[:, 1] if hasattr(rf, "predict_proba") else rf.predict(full_scaled)
    )
    full_pred = (full_prob >= threshold).astype(int)

    out = df.copy()
    out["underserved_pred_prob"] = full_prob
    out["underserved_pred"] = full_pred
    csv_bytes = out.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download CSV with Predictions",
        data=csv_bytes,
        file_name="underserved_predictions.csv",
        mime="text/csv",
    )

    # HTML report
    fair_json = json.dumps(fair_simple if "fair_simple" in locals() else {}, indent=2)
    summary = {
        "test_size": test_size,
        "class_weight": class_weight,
        "threshold": threshold,
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "smote_used": smote_used,
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "auc": None if np.isnan(metrics["auc"]) else float(metrics["auc"]),
        "fairness_json": fair_json,
    }
    html_bytes = render_html_report(summary)
    st.download_button(
        "📝 Download HTML Report",
        data=html_bytes,
        file_name="underserved_report.html",
        mime="text/html",
    )

st.markdown("---")
st.caption(
    "© 2025 — Capstone Dashboard. This template emphasizes transparency, fairness checks, threshold tuning, and exportable artifacts."
)







