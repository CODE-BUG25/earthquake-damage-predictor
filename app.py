# =====================================================
# Earthquake Damage Prediction System
# Streamlit App
# Part 1
# =====================================================



import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle

import plotly.express as px
import plotly.graph_objects as go

from pathlib import Path

# -----------------------------------------------------
# Page Configuration
# -----------------------------------------------------

st.set_page_config(
    page_title="Earthquake Damage Prediction",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------
# Custom CSS
# -----------------------------------------------------

st.markdown("""
<style>

.main{
    background-color:#F8F9FA;
}

h1,h2,h3{
    color:#0B5394;
}

.stButton>button{
    width:100%;
    background:#0B5394;
    color:white;
    border-radius:10px;
    height:3em;
    font-size:18px;
}

.stMetric{
    background:#FFFFFF;
    padding:15px;
    border-radius:12px;
    box-shadow:2px 2px 10px rgba(0,0,0,0.15);
}

</style>
""",unsafe_allow_html=True)

# -----------------------------------------------------
# Title
# -----------------------------------------------------

st.title("🌍 Earthquake Damage Prediction")

st.markdown(
"""
Predict the expected **Damage Level**
using Machine Learning.

This project uses a trained ML model
to estimate earthquake damage severity.
"""
)

# -----------------------------------------------------
# Load Models
# -----------------------------------------------------

MODEL_PATH = Path("models/final_earthquake_model.pkl")
ENCODER_PATH = Path("models/label_encoder.pkl")

@st.cache_resource
def load_model():

    model = joblib.load(MODEL_PATH)

    encoder = pickle.load(
        open(ENCODER_PATH,"rb")
    )

    return model, encoder

model, label_encoder = load_model()

# -----------------------------------------------------
# Sidebar
# -----------------------------------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(

    "Select Page",

    [

        "Prediction",

        "Project Information",

        "Dataset",

        "Model Performance"

    ]

)

st.sidebar.markdown("---")

st.sidebar.success("Model Loaded Successfully")

# -----------------------------------------------------
# Prediction Page
# -----------------------------------------------------

if page=="Prediction":

    st.header("Enter Earthquake Details")

    col1,col2=st.columns(2)

    with col1:

        magnitude=st.slider(

            "Magnitude",

            0.0,

            10.0,

            5.5,

            0.1

        )

        depth=st.number_input(

            "Depth (km)",

            value=20.0,

            min_value=0.0

        )

        cdi=st.slider(

            "CDI",

            0.0,

            12.0,

            5.0,

            0.1

        )

        mmi=st.slider(

            "MMI",

            0.0,

            12.0,

            5.0,

            0.1

        )

        sig=st.number_input(

            "Significance",

            value=400

        )

    with col2:

        tsunami=st.selectbox(

            "Tsunami",

            [0,1]

        )

        alert=st.selectbox(

            "Alert Level",

            [

                "green",

                "yellow",

                "orange",

                "red"

            ]

        )

        continent=st.selectbox(

            "Continent",

            [

                "Asia",

                "Europe",

                "North America",

                "South America",

                "Africa",

                "Australia",

                "Antarctica"

            ]

        )

        country=st.text_input(

            "Country",

            "Japan"

        )

        year=st.slider(

            "Year",

            1995,

            2023,

            2020

        )

    st.markdown("---")

    predict=st.button("Predict Damage Level")

    # -----------------------------------------------------
# Load Preprocessor
# -----------------------------------------------------

PREPROCESSOR_PATH = Path("models/preprocessor.pkl")

@st.cache_resource
def load_preprocessor():
    return joblib.load(PREPROCESSOR_PATH)

preprocessor = load_preprocessor()


# -----------------------------------------------------
# Prediction
# -----------------------------------------------------

if page == "Prediction":

    if predict:

        # --------------------------------------------
        # Feature Engineering
        # --------------------------------------------

        if magnitude < 4:
            magnitude_category = "Low"
        elif magnitude < 5:
            magnitude_category = "Moderate"
        elif magnitude < 6.5:
            magnitude_category = "High"
        else:
            magnitude_category = "Severe"

        if depth <= 70:
            depth_category = "Shallow"
        elif depth <= 300:
            depth_category = "Intermediate"
        else:
            depth_category = "Deep"

        energy_index = 10 ** (1.5 * magnitude)

        risk_score = (
            magnitude * 0.5
            + mmi * 0.2
            + cdi * 0.2
            + sig / 1000
        )

        tsunami_flag = int(tsunami)

        high_alert = 1 if alert in ["orange", "red"] else 0

        years_ago = 2026 - year

        # --------------------------------------------
        # Build Input DataFrame
        # --------------------------------------------

        input_df = pd.DataFrame({

            "magnitude":[magnitude],

            "depth":[depth],

            "cdi":[cdi],

            "mmi":[mmi],

            "sig":[sig],

            "tsunami":[tsunami],

            "alert":[alert],

            "continent":[continent],

            "country":[country],

            "Year":[year],

            "Magnitude_Category":[magnitude_category],

            "Depth_Category":[depth_category],

            "Energy_Index":[energy_index],

            "Risk_Score":[risk_score],

            "Tsunami_Flag":[tsunami_flag],

            "High_Alert":[high_alert],

            "Years_Ago":[years_ago]

        })

        # --------------------------------------------
        # Apply Preprocessing
        # --------------------------------------------

        processed_input = preprocessor.transform(input_df)

        # --------------------------------------------
        # Predict
        # --------------------------------------------

        prediction = model.predict(processed_input)

        damage_level = label_encoder.inverse_transform(prediction)[0]

        # --------------------------------------------
        # Probability
        # --------------------------------------------

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(processed_input)[0]

            confidence = np.max(probabilities) * 100

        else:

            probabilities = None

            confidence = None

        # --------------------------------------------
        # Display Results
        # --------------------------------------------

        st.markdown("---")

        st.header("Prediction Result")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(

                "Predicted Damage",

                damage_level

            )

        with col2:

            if confidence is not None:

                st.metric(

                    "Confidence",

                    f"{confidence:.2f}%"

                )

        # --------------------------------------------
        # Risk Color
        # --------------------------------------------

        if damage_level == "Low":
            st.success("Low Risk")

        elif damage_level == "Moderate":
            st.warning("Moderate Risk")

        elif damage_level == "High":
            st.error("High Risk")

        else:
            st.error("Severe Damage Expected")

        # --------------------------------------------
        # Probability Chart
        # --------------------------------------------

        if probabilities is not None:

            probability_df = pd.DataFrame({

                "Damage Level":label_encoder.classes_,

                "Probability":probabilities

            })

            fig = px.bar(

                probability_df,

                x="Damage Level",

                y="Probability",

                text="Probability",

                title="Prediction Probability"

            )

            fig.update_traces(

                texttemplate="%{text:.2f}",

                textposition="outside"

            )

            st.plotly_chart(

                fig,

                use_container_width=True

            )

        # --------------------------------------------
        # Input Summary
        # --------------------------------------------

        st.subheader("Input Summary")

        st.dataframe(

            input_df,

            use_container_width=True

        )

        # =====================================================
# DATASET PAGE
# =====================================================

elif page == "Dataset":

    st.header("📂 Earthquake Dataset")

    dataset = pd.read_csv("data/earthquake_cleaned.csv")

    st.write("Dataset Shape")

    st.write(dataset.shape)

    st.dataframe(dataset.head(20))

    st.subheader("Statistical Summary")

    st.dataframe(dataset.describe())

    st.subheader("Missing Values")

    st.dataframe(dataset.isnull().sum())

    st.subheader("Columns")

    st.write(dataset.columns.tolist())

    # ----------------------------------------

    st.subheader("Magnitude Distribution")

    fig = px.histogram(

        dataset,

        x="magnitude",

        nbins=40,

        title="Magnitude Distribution"

    )

    st.plotly_chart(fig,use_container_width=True)

    # ----------------------------------------

    st.subheader("Depth Distribution")

    fig = px.histogram(

        dataset,

        x="depth",

        nbins=40,

        title="Depth Distribution"

    )

    st.plotly_chart(fig,use_container_width=True)

    # ----------------------------------------

    st.subheader("Earthquakes by Continent")

    continent = dataset["continent"].value_counts().reset_index()

    continent.columns=["Continent","Count"]

    fig = px.bar(

        continent,

        x="Continent",

        y="Count",

        color="Count"

    )

    st.plotly_chart(fig,use_container_width=True)

    # ----------------------------------------

    if "latitude" in dataset.columns and "longitude" in dataset.columns:

        st.subheader("Earthquake Locations")

        fig = px.scatter_geo(

            dataset,

            lat="latitude",

            lon="longitude",

            color="magnitude",

            hover_name="country",

            size="magnitude",

            projection="natural earth"

        )

        st.plotly_chart(fig,use_container_width=True)



# =====================================================
# MODEL PERFORMANCE PAGE
# =====================================================

elif page=="Model Performance":

    st.header("📈 Model Performance")

    try:

        metrics = pd.read_csv("data/final_metrics.csv")

        st.dataframe(metrics)

    except:

        st.warning("Run notebook 05 first.")

    # ----------------------------------------

    try:

        importance = pd.read_csv(

            "data/final_feature_importance.csv"

        )

        fig = px.bar(

            importance.head(20),

            x="Importance",

            y="Feature",

            orientation="h",

            title="Top 20 Important Features"

        )

        st.plotly_chart(

            fig,

            use_container_width=True

        )

    except:

        st.info("Feature Importance File Missing.")

    # ----------------------------------------

    try:

        compare = pd.read_csv(

            "data/tuned_model_results.csv"

        )

        fig = px.bar(

            compare,

            x="Model",

            y="Accuracy",

            color="Accuracy",

            title="Model Comparison"

        )

        st.plotly_chart(

            fig,

            use_container_width=True

        )

    except:

        st.info("Comparison File Missing.")



# =====================================================
# PROJECT INFORMATION
# =====================================================

elif page=="Project Information":

    st.header("🌍 Earthquake Damage Prediction")

    st.markdown("""

### Project Objective

Predict earthquake damage severity using
Machine Learning.

---

### Algorithms Used

- Logistic Regression

- Decision Tree

- Random Forest

- KNN

- SVM

- Naive Bayes

- Gradient Boosting

- AdaBoost

- Extra Trees

---

### Feature Engineering

- Magnitude Category

- Depth Category

- Energy Index

- Risk Score

- Tsunami Flag

- High Alert

- Years Ago

---

### Technologies

- Python

- Pandas

- NumPy

- Scikit Learn

- Plotly

- Streamlit

- Joblib

---

### Workflow

Dataset

↓

Cleaning

↓

EDA

↓

Feature Engineering

↓

Model Building

↓

Hyperparameter Tuning

↓

Evaluation

↓

Deployment

""")



# =====================================================
# DOWNLOAD PREDICTIONS
# =====================================================

try:

    prediction_file = pd.read_csv(

        "data/test_predictions.csv"

    )

    csv = prediction_file.to_csv(index=False)

    st.download_button(

        "📥 Download Predictions",

        csv,

        file_name="predictions.csv",

        mime="text/csv"

    )

except:

    pass



# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.markdown(

"""
<div style='text-align:center'>

<h4>🌍 Earthquake Damage Prediction System</h4>

Built using

Python • Streamlit • Scikit-Learn • Plotly

Developed by Sourabh Verma

</div>
""",

unsafe_allow_html=True

)