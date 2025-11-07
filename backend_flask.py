# ===============================================
# 💰 SMB Cash Flow Risk Predictor Backend (Flask)
# ===============================================
# Author: Prasad Challa
# Description: Flask backend API that accepts a CSV,
# runs Prophet forecasting, classifies risk,
# and returns results + AI summary using OpenAI.
# ===============================================

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from prophet import Prophet
from openai import OpenAI
import io
import os

# -------------------------------------------
# CONFIGURATION
# -------------------------------------------
app = Flask(__name__)
CORS(app)  # Enable frontend access (Streamlit/Postman)

# Load API key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("⚠️ Warning: OPENAI_API_KEY not found. Set it using `setx OPENAI_API_KEY \"your-key\"`.")
else:
    print("✅ OpenAI API key detected.")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


# -------------------------------------------
# HELPER: Run Prophet Forecast
# -------------------------------------------
def run_forecast(df, months_to_forecast):
    df.columns = [c.strip().capitalize() for c in df.columns]
    df["Month"] = pd.to_datetime(df["Month"], errors="coerce")
    df = df.sort_values("Month").reset_index(drop=True)

    # Ensure required columns exist
    for col in ["Revenue", "Expenses", "Receivables", "Payables"]:
        if col not in df.columns:
            df[col] = 0

    # Calculate net cash
    df["Net_cash"] = df["Revenue"] + df["Receivables"] - (df["Expenses"] + df["Payables"])
    prophet_df = df.rename(columns={"Month": "ds", "Net_cash": "y"})

    # Build & fit Prophet model
    model = Prophet(interval_width=0.8)
    model.fit(prophet_df)

    # Forecast future months
    future = model.make_future_dataframe(periods=months_to_forecast, freq='M')
    forecast = model.predict(future).tail(months_to_forecast)

    result = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    result.columns = ["Month", "Predicted_Net_Cash", "Lower_Bound", "Upper_Bound"]
    return result


# -------------------------------------------
# HELPER: Classify Risk
# -------------------------------------------
def classify_risk(forecast_df, avg_exp):
    risks, reasons = [], []

    for _, row in forecast_df.iterrows():
        lower = row["Lower_Bound"]
        pred = row["Predicted_Net_Cash"]

        warning_threshold = 0.1 * avg_exp
        critical_threshold = -0.25 * avg_exp

        if lower <= critical_threshold:
            risk = "🔴 Critical"
            reason = f"Large shortfall likely (lower bound ₹{lower:,.0f})."
        elif lower < warning_threshold:
            risk = "🟠 Warning"
            reason = f"Cash position tight (lower bound ₹{lower:,.0f})."
        else:
            risk = "🟢 Safe"
            reason = f"No deficit expected; projected ₹{pred:,.0f}."

        risks.append(risk)
        reasons.append(reason)

    forecast_df["Risk"] = risks
    forecast_df["Reason"] = reasons
    return forecast_df


# -------------------------------------------
# ROUTE: Home
# -------------------------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "✅ SMB Cash Flow Flask Backend Running"})


# -------------------------------------------
# ROUTE: Forecast Prediction
# -------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # 1️⃣ Get uploaded file & months
        file = request.files.get("file")
        months = int(request.form.get("months", 3))

        if not file:
            return jsonify({"error": "No file uploaded"}), 400

        # 2️⃣ Read dataset
        content = file.read()
        df = pd.read_csv(io.BytesIO(content))

        # 3️⃣ Run forecast
        forecast_df = run_forecast(df, months)
        avg_exp = df["Expenses"].mean()
        forecast_df = classify_risk(forecast_df, avg_exp)

        # 4️⃣ Convert to JSON
        forecast_json = forecast_df.to_dict(orient="records")

        # 5️⃣ Generate AI Summary
        ai_summary = "⚠️ AI summary could not be generated."

        if client:
            try:
                print("🧠 Generating AI Summary...")
                table = forecast_df.to_markdown(index=False)
                prompt = f"""
                You are a financial forecasting assistant.
                Analyze this {months}-month SMB cash flow forecast:

                {table}

                Write a short, professional summary covering:
                - Risk trends (Critical/Warning/Safe)
                - Main reasons for risks
                - Actionable advice for improving liquidity
                Keep it clear and business-friendly.
                """

                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are an expert financial advisor for small businesses."},
                        {"role": "user", "content": prompt}
                    ]
                )

                ai_summary = response.choices[0].message.content
                print("✅ AI Summary generated successfully!")
                print("🧾 Summary:", ai_summary)

            except Exception as e:
                print(f"❌ AI Error: {e}")
                ai_summary = f"⚠️ AI summary failed: {e}"
        else:
            print("⚠️ No OpenAI API key found. Skipping AI summary.")
            ai_summary = "⚠️ Missing OpenAI API key."

        # 6️⃣ Return full response
        return jsonify({
            "message": f"{months}-month forecast generated successfully.",
            "forecast": forecast_json,
            "ai_summary": ai_summary
        })

    except Exception as e:
        print(f"❌ Backend error: {e}")
        return jsonify({"error": str(e)}), 500


# -------------------------------------------
# MAIN ENTRY POINT
# -------------------------------------------
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
