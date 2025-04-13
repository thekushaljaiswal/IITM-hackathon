# app.py

import streamlit as st
import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Toronto Crime Year Predictor")
st.write("Using longitude and latitude to predict the year of crime occurrence.")

@st.cache_data
def load_data():
    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "mohammadbadi/crimes-in-toronto",
        ""
    )
    df = df.dropna()
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    return df

df = load_data()

if 'longitude' in df.columns and 'latitude' in df.columns and 'occurrence_year' in df.columns:
    X = df[['longitude', 'latitude']]
    y = df['occurrence_year']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.subheader("Model Performance")
    st.write(f"**Mean Squared Error:** {mse:.2f}")
    st.write(f"**R² Score:** {r2:.4f}")

    st.subheader("Actual vs Predicted Plot")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=y_pred, ax=ax, alpha=0.6)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Actual vs Predicted Year")
    ax.grid(True)
    st.pyplot(fig)

    st.subheader("Try Prediction")
    long = st.number_input("Longitude", value=X.mean()[0])
    lat = st.number_input("Latitude", value=X.mean()[1])
    if st.button("Predict Crime Year"):
        pred_year = model.predict([[long, lat]])[0]
        st.success(f"Predicted Occurrence Year: {pred_year:.0f}")
else:
    st.error("Required columns not found in the dataset.")
