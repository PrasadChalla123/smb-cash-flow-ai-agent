# ===============================================
# 💰 SMB Cash Flow Risk Predictor Frontend (Streamlit)
# ===============================================
# Author: Prasad Challa
# Description: Streamlit frontend that connects to Flask backend
# ===============================================

import streamlit as st
import requests
import pandas as pd

# -------------------------------------------
# PAGE CONFIG
# -------------------------------------------
st.set_page_config(page_title="💰 SMB Cash Flow Predictor", layout="wide")
st.title("🤖 SMB Cash Flow Predictor – Connected to Flask API")

st.markdown("""
Upload your financial dataset, choose forecast duration, and see backend-generated insights.
""")

# -------------------------------------------
# FILE UPLOAD & INPUT
# -------------------------------------------
uploaded = st.file_uploader("📂 Upload your CSV file", type=["csv"])
months = st.number_input("📅 Enter number of months to forecast", min_value=1, max_value=12, value=3, step=1)

if uploaded is not None:
    st.write("✅ File uploaded:", uploaded.name)
    st.write("Forecast duration:", months, "months")

    if st.button("🚀 Generate Forecast"):
        with st.spinner("Sending file to Flask backend..."):
            try:
                # Send request to Flask backend
                files = {"file": uploaded.getvalue()}
                data = {"months": months}
                response = requests.post("http://127.0.0.1:5000/predict", files={"file": uploaded}, data=data)

                if response.status_code == 200:
                    result = response.json()
                    st.success(result["message"])

                    # Display forecast table
                    forecast_df = pd.DataFrame(result["forecast"])
                    st.subheader("📊 Forecast Results")
                    st.dataframe(forecast_df)

                    # Display AI summary
                    st.subheader("💬 AI Summary")
                    st.write(result["ai_summary"])

                else:
                    st.error(f"❌ Backend Error: {response.text}")

            except Exception as e:
                st.error(f"Error connecting to backend: {e}")
else:
    st.info("Please upload your CSV to start.")
