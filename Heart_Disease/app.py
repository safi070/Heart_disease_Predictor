
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="HeartWise | Heart Disease ML",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# 2. CUSTOM STYLING
# ============================================================

st.markdown(
    """
    
<style>
    /* Main page */
    .stApp {
        background: linear-gradient(145deg, #f4f7fb 0%, #eaf1f8 100%);
        color: #172b4d;
    }

    .block-container {
        max-width: 1100px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Make normal text readable */
    .stApp,
    .stApp p,
    .stApp label,
    .stApp span,
    .stApp li,
    .stApp .stMarkdown,
    .stApp [data-testid="stCaptionContainer"] {
        color: #02112b;
    }

    /* Hero banner: explicitly white text on dark background */
    .hero {
        background: linear-gradient(120deg, #102a43, #176b87);
        padding: 2rem;
        border-radius: 20px;
        color: #ffffff;
        margin-bottom: 1.4rem;
        box-shadow: 0 8px 25px rgba(16, 42, 67, 0.15);
    }

    .hero h1 {
        color: #ffffff !important;
        font-size: 2.35rem;
        margin-bottom: 0.4rem;
    }

    .hero p {
        color: #e1f2f7 !important;
        font-size: 1rem;
        margin-bottom: 0;
    }

    /* Card content */
    .section-card {
        background: #ffffff;
        color: #172b4d;
        padding: 1.4rem;
        border-radius: 16px;
        border: 1px solid #dce6ef;
        margin-bottom: 1rem;
    }

    .section-card h4,
    .section-card p {
        color: #172b4d !important;
    }

    /* Input fields */
    .stApp input,
    .stApp textarea,
    .stApp [data-baseweb="select"] > div {
        color: #172b4d !important;
        background-color: #ffffff !important;
    }

    /* Buttons */
    div.stButton > button,
    div.stFormSubmitButton > button {
        background: #176b87;
        color: #ffffff !important;
        border: none;
        border-radius: 10px;
        min-height: 3rem;
        font-weight: 600;
        width: 100%;
    }

    div.stButton > button:hover,
    div.stFormSubmitButton > button:hover {
        background: #102a43;
        color: #ffffff !important;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: #ffffff;
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid #dce6ef;
    }

    [data-testid="stMetricLabel"],
    [data-testid="stMetricValue"] {
        color: #172b4d !important;
    }

    /* Footer */
    .footer {
        color: #526579 !important;
        font-size: 0.85rem;
        text-align: center;
        padding-top: 1.5rem;
    }
</style>

    """,
    unsafe_allow_html=True,
)


# ============================================================
# 3. MODEL LOADING
# ============================================================

MODEL_PATH = Path(__file__).resolve().parent / "heart_disease_svm_model.pkl"


@st.cache_resource
def load_model(model_path: str, modified_time: float):
    """
    Load the previously trained preprocessing + SVM pipeline.
    Cache it so it is not loaded again on every interaction.
    """
    return joblib.load(model_path)


# ============================================================
# 4. HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>❤️ HeartWise</h1>
        <p>
            Heart Disease Prediction · Machine Learning Demonstration
            <br>
            Explore an SVM model trained on four features from the
            UCI Cleveland Heart Disease dataset.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Educational and medical limitation notice
st.warning(
    "**Educational use only:** This model is not a medical diagnostic "
    "tool. It has not been clinically validated and must not be used "
    "to make healthcare decisions."
)


# ============================================================
# 5. VERIFY MODEL FILE
# ============================================================

if not MODEL_PATH.is_file():
    st.error("The trained model file could not be found.")
    st.write("Expected model location:")
    st.code(str(MODEL_PATH))
    st.info(
        "Place `heart_disease_svm_model.pkl` in the same folder "
        "as `app.py`, then restart the application."
    )
    st.stop()

try:
    model = load_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
except Exception as exc:
    st.error("The model could not be loaded.")
    st.caption(f"Technical details: {type(exc).__name__}: {exc}")
    st.info(
        "Check that the model file is valid and that the deployed "
        "scikit-learn version is compatible with the training version."
    )
    st.stop()


# ============================================================
# 6. MODEL INFORMATION
# ============================================================

metric_col1, metric_col2, metric_col3 = st.columns(3)

with metric_col1:
    st.metric("Model", "SVM / SVC")

with metric_col2:
    st.metric("Input Features", "4")

with metric_col3:
    st.metric("Test Accuracy", "78.69%")

st.caption(
    "Test accuracy is from this project's 61-record test split. "
    "It is not a measure of clinical reliability."
)

st.divider()


# ============================================================
# 7. INPUT FORM
# ============================================================

left_col, right_col = st.columns([1.15, 0.85], gap="large")

with left_col:
    st.subheader("Enter the input features")
    st.write(
        "Provide the four values required by the trained model."
    )

    with st.form("heart_prediction_form"):
        age = st.number_input(
            "Age (years)",
            min_value=1,
            max_value=120,
            value=50,
            step=1,
            help="Age in years.",
        )

        sex_label = st.selectbox(
            "Sex (original dataset encoding)",
            options=["Male", "Female"],
            help="The Cleveland dataset encodes male as 1 and female as 0.",
        )
        sex = 1 if sex_label == "Male" else 0

        cp_labels = {
            "1 — Typical angina": 1,
            "2 — Atypical angina": 2,
            "3 — Non-anginal pain": 3,
            "4 — Asymptomatic": 4,
        }

        cp_label = st.selectbox(
            "Chest pain type",
            options=list(cp_labels.keys()),
            help=(
                "Select a category using the original UCI dataset's "
                "numeric codes."
            ),
        )
        cp = cp_labels[cp_label]

        thalach = st.number_input(
            "Maximum heart rate achieved",
            min_value=50,
            max_value=250,
            value=150,
            step=1,
            help="Enter a value in beats per minute.",
        )

        submitted = st.form_submit_button(
            "Run Prediction",
            use_container_width=True,
        )

with right_col:
    st.subheader("About this model")

    st.markdown(
        """
        <div class="section-card">
            <h4>🧠 Machine learning</h4>
            <p>
                The application uses a Support Vector Machine (SVM)
                classifier trained on the Cleveland dataset.
            </p>
            <hr>
            <h4>📋 Four input features</h4>
            <p>
                Age · Sex · Chest pain type · Maximum heart rate
            </p>
            <hr>
            <h4>⚙️ Prediction classes</h4>
            <p>
                <b>Class 0:</b> No disease indicated in the dataset<br>
                <b>Class 1:</b> Disease indicated in the dataset
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# 8. PREDICTION
# ============================================================

if submitted:
    input_data = pd.DataFrame(
        [
            {
                "age": float(age),
                "sex": float(sex),
                "cp": float(cp),
                "thalach": float(thalach),
            }
        ],
        columns=["age", "sex", "cp", "thalach"],
    )

    st.divider()
    st.subheader("Prediction result")

    try:
        prediction = int(model.predict(input_data)[0])

        if prediction == 1:
            st.error(
                "### Class 1 — Disease indicated\n"
                "The model classified this input into the "
                "'disease indicated' category."
            )
        else:
            st.success(
                "### Class 0 — No disease indicated\n"
                "The model classified this input into the "
                "'no disease indicated' category."
            )

        with st.expander("View submitted input values"):
            st.dataframe(input_data, use_container_width=True)

        st.caption(
            "This is a machine-learning classification, not a "
            "diagnosis. A result in either class cannot establish "
            "whether an individual has or does not have heart disease. "
            "Consult a qualified healthcare professional for medical concerns."
        )

    except Exception as exc:
        st.error("Prediction failed. Please check the model and inputs.")
        st.caption(f"Technical details: {type(exc).__name__}: {exc}")


# ============================================================
# 9. FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        Academic Machine Learning Project · UCI Cleveland Heart Disease Dataset
        <br>
        Educational demonstration only · Not for clinical use
    </div>
    """,
    unsafe_allow_html=True,
)
