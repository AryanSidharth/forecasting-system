import streamlit as st
import requests
import pandas as pd
import plotly.express as px


st.set_page_config(
    page_title="Forecasting Platform",
    page_icon="📈",
    layout="wide"
)


# =========================
# CUSTOM CSS
# =========================

st.markdown(
    '''
    <style>
    .main {
        background-color: #0E1117;
        color: white;
    }

    .metric-card {
        background-color: #1E1E1E;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
    }

    h1, h2, h3 {
        color: white;
    }
    </style>
    ''',
    unsafe_allow_html=True
)


# =========================
# HEADER
# =========================

st.title("📈 AI Forecasting Platform")

st.markdown(
    """
    Production-ready forecasting system using:
    - XGBoost
    - SARIMA
    - Prophet
    - LSTM
    """
)


# =========================
# STATE SELECTOR
# =========================

states = [
    'California',
    'Texas',
    'Florida',
    'New York',
    'Georgia',
    'Illinois',
    'Arizona',
    'Colorado'
]

selected_state = st.selectbox(
    "Select State",
    states
)


# =========================
# API CALL
# =========================

if st.button("Generate Forecast"):

    with st.spinner("Generating Forecast..."):

        url = f"http://127.0.0.1:8000/predict?state={selected_state}"

        response = requests.get(url)

        data = response.json()

        if "8_week_forecast" in data:

            forecast = data["8_week_forecast"]

            # =========================
            # METRICS
            # =========================

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Forecast Weeks",
                len(forecast)
            )

            col2.metric(
                "Max Forecast",
                f"{max(forecast):,.0f}"
            )

            col3.metric(
                "Min Forecast",
                f"{min(forecast):,.0f}"
            )

            # =========================
            # DATAFRAME
            # =========================

            forecast_df = pd.DataFrame({
                "Week": [f"Week {i+1}" for i in range(8)],
                "Forecast": forecast
            })

            st.subheader("Forecast Data")

            st.dataframe(
                forecast_df,
                use_container_width=True
            )

            # =========================
            # CHART
            # =========================

            fig = px.line(
                forecast_df,
                x="Week",
                y="Forecast",
                markers=True,
                title=f"8 Week Forecast for {selected_state}"
            )

            fig.update_layout(
                template="plotly_dark",
                height=500
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:

            st.error(data["error"])