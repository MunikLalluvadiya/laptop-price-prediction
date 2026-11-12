
import json
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path


# ==================================
# 1. PAGE CONFIGURATION
# ==================================

st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="centered"
)


# ==================================
# 2. PROJECT PATHS
# ==================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"


# ==================================
# 3. LOAD MODEL AND JSON FILES
# ==================================

@st.cache_resource
def load_assets():

    model = joblib.load(
        MODEL_DIR / "tuned_xgb_model.pkl"
    )

    with open(MODEL_DIR / "feature_columns.json", "r") as f:
        feature_columns = json.load(f)

    with open(MODEL_DIR / "preprocessing.json", "r") as f:
        preprocessing = json.load(f)

    return model, feature_columns, preprocessing


model, feature_columns, preprocessing = load_assets()


# ==================================
# 4. CATEGORY OPTIONS
# ==================================

def get_options(column):

    keep_list = preprocessing[f"{column.lower()}_keep"]
    categories = preprocessing["categories"][column]

    options = [
        value for value in keep_list
        if value in categories
    ]

    if "Other" in categories:
        options.append("Other")

    return list(dict.fromkeys(options))


# Check Other category availability
for col in ["Brand", "CPU", "GPU"]:

    if "Other" not in preprocessing["categories"][col]:
        st.error(f"'Other' category missing for {col}.")
        st.stop()


# ==================================
# 5. APP TITLE
# ==================================

st.title("💻 Laptop Price Predictor")

st.write(
    "Enter your laptop specifications "
    "to estimate its price in Euros."
)


# ==================================
# 6. USER INPUTS
# ==================================

col1, col2 = st.columns(2)


with col1:

    Brand = st.selectbox(
        "Brand",
        get_options("Brand")
    )

    CPU = st.selectbox(
        "CPU",
        get_options("CPU")
    )

    RAM = st.number_input(
        "RAM (GB)",
        min_value=1,
        value=16
    )

    Storage = st.number_input(
        "Storage (GB)",
        min_value=1,
        value=512
    )

    GPU = st.selectbox(
        "GPU",
        get_options("GPU")
    )


with col2:

    Screen = st.number_input(
        "Screen Size (inches)",
        min_value=1.0,
        value=15.6,
        step=0.1
    )

    Status = st.selectbox(
        "Status",
        preprocessing["categories"]["Status"]
    )

    Storage_type = st.selectbox(
        "Storage Type",
        preprocessing["categories"]["Storage type"]
    )

    Touch = st.selectbox(
        "Touch",
        preprocessing["categories"]["Touch"]
    )


# ==================================
# 7. PREDICTION BUTTON
# ==================================

if st.button("Predict Price", type="primary"):

    # Create input DataFrame
    input_data = pd.DataFrame([{
        "Status": Status,
        "Brand": Brand,
        "CPU": CPU,
        "Storage type": Storage_type,
        "GPU": GPU,
        "Touch": Touch,
        "RAM": RAM,
        "Storage": Storage,
        "Screen": Screen
    }])


    # ==================================
    # 8. APPLY TRAINING CATEGORIES
    # ==================================

    categorical_columns = [
        "Status",
        "Brand",
        "CPU",
        "Storage type",
        "GPU",
        "Touch"
    ]

    for col in categorical_columns:

        input_data[col] = pd.Categorical(
            input_data[col],
            categories=preprocessing["categories"][col]
        )


    # Check invalid categories
    if input_data[categorical_columns].isna().any().any():

        st.error("Invalid categorical input.")
        st.stop()


    # ==================================
    # 9. ONE-HOT ENCODING
    # ==================================

    input_encoded = pd.get_dummies(
        input_data,
        columns=categorical_columns,
        drop_first=False,
        dtype=int
    )


    # ==================================
    # 10. FEATURE ALIGNMENT
    # ==================================

    # Check missing training columns
    missing_columns = [
        col for col in feature_columns
        if col not in input_encoded.columns
    ]

    if missing_columns:

        st.error(
            f"Missing training features: {missing_columns}"
        )
        st.stop()


    # Align exact training feature order
    input_encoded = input_encoded.reindex(
        columns=feature_columns
    )


    # Final safety check
    if input_encoded.isna().any().any():

        st.error("Missing values found in encoded input.")
        st.stop()


    # ==================================
    # 11. MODEL PREDICTION
    # ==================================

    prediction = model.predict(input_encoded)[0]


    # ==================================
    # 12. LOG TO EURO CONVERSION
    # ==================================

    predicted_price = np.expm1(prediction)


    # ==================================
    # 13. DISPLAY RESULT
    # ==================================

    st.success("Prediction Successful!")

    st.metric(
        label="Predicted Laptop Price",
        value=f"€{predicted_price:,.2f}"
    )

    st.caption(
        "Model test-set MAE: approximately €224. "
        "This is an average absolute error, not a guaranteed range."
    )

    st.info(
        "Model limitation: Estimates below €1,500 have been "
        "more accurate in your evaluation. Premium laptops "
        "may be underpredicted."
    )
