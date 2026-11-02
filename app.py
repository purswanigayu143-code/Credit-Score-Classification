
from pathlib import Path
import warnings
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

warnings.filterwarnings("ignore")

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="CreditIQ | Credit Score Classification",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- FILE PATHS ----------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "credit_score_model.joblib"
RESULTS_PATH = BASE_DIR / "model_comparison.xls"
IMPORTANCE_PATH = BASE_DIR / "feature_importance.xls"

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: Inter, sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #f7faff, #eef4ff 55%, #f9fbff);
}

.block-container {
    padding-top: 1.5rem;
    max-width: 1500px;
}

[data-testid="stSidebar"] {
    background: white;
    border-right: 1px solid #e5eaf4;
}

.hero {
    background: linear-gradient(115deg, #102a56, #1f5eb5 60%, #3b82df);
    padding: 30px 34px;
    border-radius: 22px;
    color: white;
    box-shadow: 0 14px 34px #18468725;
    margin-bottom: 24px;
}

.hero h1 {
    font-size: 2.2rem;
    font-weight: 800;
    margin: 0 0 8px;
    color: white;
}

.hero p {
    font-size: 1rem;
    margin: 0;
    color: #eaf2ff;
}

.metric-card {
    background: white;
    border: 1px solid #e2eaf6;
    border-radius: 16px;
    padding: 18px 20px;
    box-shadow: 0 5px 18px #1e3c640b;
    min-height: 100px;
}

.metric-label {
    font-size: .88rem;
    color: #65758b;
    margin-bottom: 8px;
}

.metric-value {
    font-size: 1.7rem;
    font-weight: 750;
    color: #152d50;
    line-height: 1.2;
}

.section-title {
    font-size: 1.3rem;
    font-weight: 750;
    color: #18345d;
    margin: 8px 0 14px;
}
</style>
""", unsafe_allow_html=True)


# ---------------- MODEL LOADING ----------------
@st.cache_resource
def load_model(path, modified):
    return joblib.load(path)


def load_saved_model():
    if not MODEL_PATH.exists():
        return None, f"Model file not found: {MODEL_PATH.name}"

    try:
        model = load_model(
            str(MODEL_PATH),
            MODEL_PATH.stat().st_mtime
        )
        return model, None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


model, model_error = load_saved_model()


# ---------------- MODEL HELPERS ----------------
def get_steps(est):
    return getattr(est, "named_steps", {}) or {}


def get_preprocessor(est):
    for _, step in get_steps(est).items():
        if hasattr(step, "transformers_") or hasattr(step, "transformers"):
            return step
    return None


def get_features(est):
    if hasattr(est, "feature_names_in_"):
        return list(est.feature_names_in_)

    prep = get_preprocessor(est)

    if prep is not None and hasattr(prep, "feature_names_in_"):
        return list(prep.feature_names_in_)

    found = []

    if prep is not None:
        for _, transformer, cols in getattr(prep, "transformers_", []):
            if isinstance(cols, (list, tuple, np.ndarray, pd.Index)):
                found.extend(list(cols))

    if found:
        return list(dict.fromkeys(map(str, found)))

    for step in reversed(list(get_steps(est).values())):
        if hasattr(step, "feature_names_in_"):
            return list(step.feature_names_in_)

    return []


def get_types(est, features):
    prep = get_preprocessor(est)
    numeric, categorical = [], []

    if prep is not None:
        for name, transformer, cols in getattr(prep, "transformers_", []):
            if not isinstance(cols, (list, tuple, np.ndarray, pd.Index)):
                continue

            cols = list(cols)
            lname = str(name).lower()
            tname = str(transformer).lower()

            if "num" in lname:
                numeric.extend(cols)
            elif "cat" in lname or "onehot" in tname:
                categorical.extend(cols)
            elif "standardscaler" in tname or "simpleimputer" in tname:
                numeric.extend(cols)

    numeric = list(dict.fromkeys(f for f in numeric if f in features))
    categorical = list(dict.fromkeys(f for f in categorical if f in features))

    unknown = [
        f for f in features
        if f not in numeric and f not in categorical
    ]

    return numeric, categorical, unknown


def get_options(est, feature):
    prep = get_preprocessor(est)

    if prep is None:
        return []

    for _, transformer, cols in getattr(prep, "transformers_", []):
        if not isinstance(cols, (list, tuple, np.ndarray, pd.Index)):
            continue

        cols = list(cols)

        if feature not in cols:
            continue

        candidates = [transformer]

        if hasattr(transformer, "named_steps"):
            candidates += list(transformer.named_steps.values())

        for item in candidates:
            if hasattr(item, "categories_"):
                try:
                    index = cols.index(feature)
                    return [
                        str(v)
                        for v in item.categories_[index]
                        if pd.notna(v)
                    ]
                except Exception:
                    pass

    return []


def get_classes(est):
    if hasattr(est, "classes_"):
        return [str(v) for v in est.classes_]

    for step in reversed(list(get_steps(est).values())):
        if hasattr(step, "classes_"):
            return [str(v) for v in step.classes_]

    return []


# ---------------- EXCEL LOADING ----------------
def load_excel_file(path):
    if not path.exists():
        return None, f"{path.name} not found"

    # First try as an Excel file; legacy .xls requires xlrd.
    try:
        return pd.read_excel(path), None
    except Exception as excel_error:
        # Some files have CSV text but an .xls extension.
        try:
            return pd.read_csv(path), None
        except Exception:
            return None, (
                f"Could not read {path.name}. "
                f"Check whether it is a valid Excel/CSV file. "
                f"Details: {excel_error}"
            )


results_df, results_error = load_excel_file(RESULTS_PATH)
importance_df, importance_error = load_excel_file(IMPORTANCE_PATH)

features = get_features(model) if model is not None else []

if model is not None and features:
    numeric_features, categorical_features, unknown_features = get_types(
        model, features
    )
else:
    numeric_features, categorical_features, unknown_features = [], [], []

classes = get_classes(model) if model is not None else []


# ---------------- UI HELPERS ----------------
def hero(title, subtitle):
    st.markdown(
        f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True
    )


def metric(label, value, hint=""):
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>'
        f'<div style="color:#7b8798;font-size:.82rem">{hint}</div>'
        f'</div>',
        unsafe_allow_html=True
    )


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("## 💳 CreditIQ")
    st.caption("CREDIT SCORE INTELLIGENCE")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "Credit Score Prediction",
            "Model Performance",
            "Data Explorer",
            "About Project"
        ]
    )

    st.divider()
    st.markdown("### System Status")

    if model is not None:
        st.success("Prediction model loaded")
    else:
        st.error("Prediction model unavailable")

    if model_error:
        st.caption(model_error)

    st.caption("Machine Learning | Classification")
    st.caption("Educational project dashboard")


# =========================================================
# OVERVIEW
# =========================================================
if page == "Overview":
    hero(
        "Credit Score Intelligence",
        "An interactive machine learning dashboard for credit score "
        "classification, model evaluation and data exploration."
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric(
            "Model Status",
            "Loaded" if model is not None else "Not Loaded",
            "Saved estimator"
        )

    with c2:
        metric(
            "Models Compared",
            str(len(results_df)) if results_df is not None else "N/A",
            "Rows in comparison sheet"
        )

    with c3:
        metric(
            "Input Features",
            str(len(features)) if features else "Needs setup",
            "Detected from fitted model"
        )

    with c4:
        metric(
            "Target Classes",
            str(len(classes)) if classes else "N/A",
            "Credit score categories"
        )

    st.divider()

    left, right = st.columns([1.25, 1])

    with left:
        st.markdown(
            '<div class="section-title">📊 Model Comparison</div>',
            unsafe_allow_html=True
        )

        if results_df is not None and not results_df.empty:
            st.dataframe(
                results_df,
                use_container_width=True,
                hide_index=True
            )

            model_col = next(
                (
                    c for c in results_df.columns
                    if "model" in c.lower() or "algorithm" in c.lower()
                ),
                results_df.columns[0]
            )

            nums = results_df.select_dtypes(
                include=np.number
            ).columns.tolist()

            metric_col = next(
                (
                    c for c in nums
                    if any(
                        k in c.lower()
                        for k in ["macro f1", "f1", "accuracy", "score"]
                    )
                ),
                None
            )

            if metric_col:
                fig = px.bar(
                    results_df,
                    x=model_col,
                    y=metric_col,
                    title=f"{metric_col} by model",
                    text_auto=".3f"
                )
                fig.update_layout(template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(results_error or "Model comparison sheet unavailable.")

    with right:
        st.markdown(
            '<div class="section-title">🧭 Project Workflow</div>',
            unsafe_allow_html=True
        )

        workflow = [
            ("01 — Data Preparation", "Clean data and handle missing values."),
            ("02 — Model Training", "Train and compare classification models."),
            ("03 — Model Evaluation", "Review available metrics."),
            ("04 — Prediction", "Estimate a credit score category.")
        ]

        for title, description in workflow:
            st.markdown(f"**{title}**")
            st.write(description)

    if model is not None and not features:
        st.warning(
            "Model loaded, but original input feature names were not "
            "recovered. See Prediction page diagnostics."
        )


# =========================================================
# CREDIT SCORE PREDICTION
# =========================================================
elif page == "Credit Score Prediction":
    hero(
        "Credit Score Prediction",
        "Enter customer details to estimate a credit score category "
        "using your saved model."
    )

    if model is None:
        st.error(model_error or "Model unavailable.")

    elif not features:
        st.error(
            "The model loaded, but original input feature names "
            "could not be identified."
        )

        st.write(
            "The saved model may be a classifier only rather than "
            "the complete fitted preprocessing pipeline."
        )

        with st.expander("Developer diagnostics"):
            st.write("Model type:", str(type(model)))
            st.write("Pipeline steps:", list(get_steps(model).keys()))
            st.write(
                "Preprocessor found:",
                get_preprocessor(model) is not None
            )
            st.write("Target classes:", classes)
            st.write("Expected file:", str(MODEL_PATH))

        st.info(
            "Export the complete fitted preprocessing pipeline and "
            "original input feature names from the training notebook. "
            "Do not guess the feature columns."
        )

    elif unknown_features:
        st.warning(
            "Some feature types could not be recovered from the saved model."
        )
        st.write("Unclassified features:", unknown_features)

        st.info(
            "Check the fitted preprocessing pipeline before creating "
            "prediction inputs."
        )

    else:
        st.markdown("### 🧾 Customer Profile")
        st.caption(f"Model expects {len(features)} input features.")

        with st.form("prediction_form"):
            values = {}
            cols = st.columns(3)

            for i, feature in enumerate(features):
                with cols[i % 3]:
                    label = str(feature).replace("_", " ").title()

                    if feature in numeric_features:
                        values[feature] = st.number_input(
                            label,
                            value=0.0,
                            step=1.0,
                            format="%.4f"
                        )

                    elif feature in categorical_features:
                        options = get_options(model, feature)

                        if options:
                            values[feature] = st.selectbox(
                                label,
                                options
                            )
                        else:
                            values[feature] = st.text_input(label)

            submitted = st.form_submit_button(
                "✨ Predict Credit Score",
                type="primary",
                use_container_width=True
            )

        if submitted:
            try:
                row = pd.DataFrame(
                    [{f: values[f] for f in features}],
                    columns=features
                )

                for feature in categorical_features:
                    row[feature] = row[feature].astype(str)

                prediction = model.predict(row)[0]

                st.markdown("### 🎯 Prediction Result")
                st.success(
                    f"Predicted Credit Score Category: **{prediction}**"
                )

                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(row)[0]

                    labels = (
                        classes
                        if len(classes) == len(probabilities)
                        else [str(i) for i in range(len(probabilities))]
                    )

                    probability_df = pd.DataFrame({
                        "Credit Score Category": labels,
                        "Probability": probabilities
                    }).sort_values(
                        "Probability",
                        ascending=False
                    )

                    st.metric(
                        "Highest model probability",
                        f"{probability_df.iloc[0]['Probability']:.1%}"
                    )

                    fig = px.bar(
                        probability_df,
                        x="Credit Score Category",
                        y="Probability",
                        text=probability_df["Probability"].map(
                            lambda x: f"{x:.1%}"
                        ),
                        title="Predicted class probabilities"
                    )

                    fig.update_layout(
                        template="plotly_white",
                        yaxis_tickformat=".0%",
                        yaxis_range=[0, 1]
                    )

                    st.plotly_chart(fig, use_container_width=True)

                st.caption(
                    "Educational estimate only; not an official credit "
                    "bureau score or lending decision."
                )

            except Exception as e:
                st.error(
                    "Prediction failed. Check that input values and "
                    "columns match the training pipeline."
                )
                st.exception(e)


# =========================================================
# MODEL PERFORMANCE
# =========================================================
elif page == "Model Performance":
    hero(
        "Model Performance",
        "Review evaluation results saved from your training notebook."
    )

    if results_df is None or results_df.empty:
        st.warning(results_error or "No model comparison table available.")
    else:
        st.dataframe(
            results_df,
            use_container_width=True,
            hide_index=True
        )

        numeric_columns = results_df.select_dtypes(
            include=np.number
        ).columns.tolist()

        model_col = next(
            (
                c for c in results_df.columns
                if "model" in c.lower() or "algorithm" in c.lower()
            ),
            results_df.columns[0]
        )

        if numeric_columns:
            metric_col = st.selectbox(
                "Choose evaluation metric",
                numeric_columns
            )

            fig = px.bar(
                results_df,
                x=model_col,
                y=metric_col,
                color=model_col,
                text_auto=".3f",
                title=f"Model comparison by {metric_col}"
            )

            fig.update_layout(
                template="plotly_white",
                showlegend=False
            )

            st.plotly_chart(fig, use_container_width=True)

        st.download_button(
            "⬇️ Download results as CSV",
            results_df.to_csv(index=False).encode("utf-8"),
            "model_comparison.csv",
            "text/csv"
        )

    if importance_df is not None and not importance_df.empty:
        st.markdown("### Feature Importance")

        st.dataframe(
            importance_df,
            use_container_width=True,
            hide_index=True
        )

        feature_col = next(
            (
                c for c in importance_df.columns
                if "feature" in c.lower() or "name" in c.lower()
            ),
            importance_df.columns[0]
        )

        value_col = next(
            (
                c for c in importance_df.columns
                if "importance" in c.lower()
                or "score" in c.lower()
                or "value" in c.lower()
            ),
            None
        )

        if value_col:
            plot_df = importance_df.copy()
            plot_df[value_col] = pd.to_numeric(
                plot_df[value_col],
                errors="coerce"
            )
            plot_df = plot_df.dropna(subset=[value_col])
            plot_df = plot_df.sort_values(
                value_col,
                ascending=False
            ).head(15)

            if not plot_df.empty:
                fig = px.bar(
                    plot_df,
                    x=value_col,
                    y=feature_col,
                    orientation="h",
                    title="Top 15 Feature Importance"
                )

                fig.update_layout(
                    template="plotly_white",
                    yaxis={"categoryorder": "total ascending"}
                )

                st.plotly_chart(fig, use_container_width=True)

    if results_error:
        st.caption(results_error)

    if importance_error:
        st.caption(importance_error)


# =========================================================
# DATA EXPLORER
# =========================================================
elif page == "Data Explorer":
    hero(
        "Data Explorer",
        "Upload a CSV dataset to inspect columns, missing values "
        "and numeric distributions."
    )

    uploaded = st.file_uploader(
        "Upload train.csv or test.csv",
        type=["csv"]
    )

    if uploaded:
        try:
            df = pd.read_csv(uploaded)

            a, b, c = st.columns(3)

            with a:
                metric("Rows", f"{len(df):,}")

            with b:
                metric("Columns", f"{df.shape[1]:,}")

            with c:
                metric(
                    "Missing Values",
                    f"{int(df.isna().sum().sum()):,}"
                )

            st.markdown("### Dataset Preview")
            st.dataframe(
                df.head(100),
                use_container_width=True
            )

            with st.expander("Summary Statistics"):
                st.dataframe(
                    df.describe(include="all").T,
                    use_container_width=True
                )

            missing = df.isna().sum().sort_values(ascending=False)
            missing = missing[missing > 0]

            if not missing.empty:
                fig = px.bar(
                    x=missing.index,
                    y=missing.values,
                    labels={
                        "x": "Column",
                        "y": "Missing Values"
                    },
                    title="Missing Values by Column"
                )

                st.plotly_chart(fig, use_container_width=True)

            numeric = df.select_dtypes(
                include=np.number
            ).columns.tolist()

            if numeric:
                chosen = st.selectbox(
                    "Numeric feature distribution",
                    numeric
                )

                fig = px.histogram(
                    df,
                    x=chosen,
                    marginal="box",
                    title=f"Distribution of {chosen}"
                )

                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Unable to read CSV: {e}")

    else:
        st.info(
            "Upload a CSV file to explore it. "
            "Files are processed locally by this app."
        )


# =========================================================
# ABOUT PROJECT
# =========================================================
elif page == "About Project":
    hero(
        "About CreditIQ",
        "A machine-learning classification project to estimate "
        "credit score categories."
    )

    st.markdown("### Project Objective")

    st.write(
        "Use a trained classification model to estimate the "
        "Credit_Score category from customer input features."
    )

    cols = st.columns(4)

    tools = [
        ("🐍", "Python"),
        ("📊", "Pandas / NumPy"),
        ("🧠", "Scikit-learn"),
        ("🚀", "Streamlit")
    ]

    for col, (emoji, name) in zip(cols, tools):
        with col:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-value">{emoji}</div>'
                f'<b>{name}</b></div>',
                unsafe_allow_html=True
            )

    st.warning(
        "Educational project only. Predictions are not official "
        "credit bureau scores or financial advice."
    )

    with st.expander("Developer Diagnostics"):
        st.write("Project folder:", str(BASE_DIR))
        st.write("Model path:", str(MODEL_PATH))
        st.write("Model exists:", MODEL_PATH.exists())

        st.write(
            "Model type:",
            str(type(model)) if model is not None else "Not loaded"
        )

        st.write(
            "Pipeline steps:",
            list(get_steps(model).keys()) if model is not None else []
        )

        st.write("Features detected:", features)
        st.write("Numeric features:", numeric_features)
        st.write("Categorical features:", categorical_features)
        st.write("Unknown features:", unknown_features)
        st.write("Comparison file:", results_error or "Loaded")
        st.write("Importance file:", importance_error or "Loaded")

        if model_error:
            st.error(model_error)


# ---------------- FOOTER ----------------
st.divider()
st.caption(
    "CreditIQ • Credit Score Classification • Educational ML Project"
)
