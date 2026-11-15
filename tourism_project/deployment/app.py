"""
Streamlit app - Wellness Tourism Package purchase prediction ("Visit with Us").

Loads the model that the GitHub Actions pipeline trained and committed to
tourism_project/deployment/, collects the customer details, stores them in a
pandas DataFrame and returns the purchase probability and the recommendation.
"""
import json
import os

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_tourism_model_v1.joblib")
CONFIG_PATH = os.path.join(BASE_DIR, "model_config.json")

FEATURE_ORDER = [
    "Age", "CityTier", "DurationOfPitch", "NumberOfPersonVisiting",
    "NumberOfFollowups", "PreferredPropertyStar", "NumberOfTrips", "Passport",
    "PitchSatisfactionScore", "OwnCar", "NumberOfChildrenVisiting", "MonthlyIncome",
    "TypeofContact", "Occupation", "Gender", "ProductPitched",
    "MaritalStatus", "Designation",
]

st.set_page_config(page_title="Wellness Package Predictor", page_icon="🧳", layout="wide")


@st.cache_resource
def load_artifacts():
    """Load the trained pipeline and its configuration (cached across reruns)."""
    model = joblib.load(MODEL_PATH)
    config = {"decision_threshold": 0.5}
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as f:
            config.update(json.load(f))
    return model, config


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🧳 Wellness Tourism Package - Purchase Prediction")
st.write(
    "Enter the customer's profile and sales-interaction details to estimate the "
    "probability that they will buy the **Wellness Tourism Package** before the "
    "sales team contacts them."
)

if not os.path.exists(MODEL_PATH):
    st.error(
        "Model file not found. Run the GitHub Actions pipeline so it commits "
        "`best_tourism_model_v1.joblib` to `tourism_project/deployment/`."
    )
    st.stop()

model, config = load_artifacts()
threshold = float(config.get("decision_threshold", 0.5))

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
with st.form("customer_form"):
    st.subheader("Customer details")
    c1, c2, c3 = st.columns(3)
    with c1:
        Age = st.number_input("Age", min_value=18, max_value=80, value=35, step=1)
        Gender = st.selectbox("Gender", ["Male", "Female"])
        MaritalStatus = st.selectbox("Marital status", ["Married", "Single", "Unmarried", "Divorced"])
        CityTier = st.selectbox("City tier", [1, 2, 3], help="Tier 1 > Tier 2 > Tier 3")
    with c2:
        Occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
        Designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
        MonthlyIncome = st.number_input("Gross monthly income", min_value=1000.0, max_value=100000.0,
                                        value=23000.0, step=500.0)
        Passport = st.radio("Valid passport?", ["Yes", "No"], horizontal=True)
    with c3:
        OwnCar = st.radio("Owns a car?", ["Yes", "No"], horizontal=True)
        NumberOfTrips = st.number_input("Average trips per year", min_value=0, max_value=25, value=3)
        NumberOfPersonVisiting = st.number_input("People on the trip", min_value=1, max_value=6, value=3)
        NumberOfChildrenVisiting = st.number_input("Children under 5 on the trip", min_value=0, max_value=4, value=1)
        PreferredPropertyStar = st.selectbox("Preferred hotel rating (stars)", [3, 4, 5])

    st.subheader("Sales interaction")
    d1, d2, d3 = st.columns(3)
    with d1:
        TypeofContact = st.selectbox("Type of contact", ["Self Enquiry", "Company Invited"])
        ProductPitched = st.selectbox("Product pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
    with d2:
        DurationOfPitch = st.number_input("Pitch duration (minutes)", min_value=1, max_value=130, value=15)
        NumberOfFollowups = st.number_input("Number of follow-ups", min_value=0, max_value=10, value=4)
    with d3:
        PitchSatisfactionScore = st.slider("Pitch satisfaction score", 1, 5, 3)

    submitted = st.form_submit_button("Predict", type="primary")

# ---------------------------------------------------------------------------
# Inputs -> DataFrame -> prediction
# ---------------------------------------------------------------------------
input_df = pd.DataFrame([{
    "Age": float(Age),
    "CityTier": float(CityTier),
    "DurationOfPitch": float(DurationOfPitch),
    "NumberOfPersonVisiting": float(NumberOfPersonVisiting),
    "NumberOfFollowups": float(NumberOfFollowups),
    "PreferredPropertyStar": float(PreferredPropertyStar),
    "NumberOfTrips": float(NumberOfTrips),
    "Passport": 1.0 if Passport == "Yes" else 0.0,
    "PitchSatisfactionScore": float(PitchSatisfactionScore),
    "OwnCar": 1.0 if OwnCar == "Yes" else 0.0,
    "NumberOfChildrenVisiting": float(NumberOfChildrenVisiting),
    "MonthlyIncome": float(MonthlyIncome),
    "TypeofContact": TypeofContact,
    "Occupation": Occupation,
    "Gender": Gender,
    "ProductPitched": ProductPitched,
    "MaritalStatus": MaritalStatus,
    "Designation": Designation,
}])[FEATURE_ORDER]

if submitted:
    prob = float(model.predict_proba(input_df)[0, 1])
    will_buy = prob >= threshold

    st.subheader("Result")
    m1, m2 = st.columns(2)
    m1.metric("Purchase probability", f"{prob:.1%}")
    m2.metric("Decision threshold", f"{threshold:.2f}")
    if will_buy:
        st.success("✅ This customer is **likely to purchase** the package - prioritise the contact.")
    else:
        st.warning("⚠️ This customer is **unlikely to purchase** the package - lower contact priority.")

    with st.expander("Input data sent to the model"):
        st.dataframe(input_df, use_container_width=True)

with st.sidebar:
    st.header("Model info")
    st.write("Algorithm: XGBoost pipeline (scaling + one-hot encoding)")
    st.write(f"Decision threshold: **{threshold:.2f}**")
    if "test_metrics" in config:
        st.write("Hold-out test metrics:")
        st.json(config["test_metrics"])
