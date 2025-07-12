from flask import Flask, request, jsonify
from flask_cors import CORS
from joblib import load
import yfinance as yf
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

# Load models
clf = load("model_direction.pkl")
reg = load("model_movement.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    symbol = data.get("symbol", "").upper()

    try:
        df = yf.download(symbol, period="7d", interval="1d")
        if df.empty:
            return jsonify({"error": "Invalid or unsupported stock symbol."}), 400

        df.dropna(inplace=True)
        X = df[['Open', 'High', 'Low', 'Close', 'Volume']].iloc[-1:]

        current_price = float(df['Close'].iloc[-1])
        direction_probs = clf.predict_proba(X)[0]  # returns [P(Down), P(Up)]
        direction = clf.predict(X)[0]              # 0 or 1
        confidence = direction_probs[direction] * 100  # Confidence for predicted class
        movement = abs(float(reg.predict(X)[0]))       # Ensure movement is positive


        # ✅ Apply direction to calculate predicted price
        if direction == 1:  # Up
            predicted_price = current_price + movement
        else:               # Down
            predicted_price = current_price - movement


        # print(movement, current_price, direction, predicted_price)

        result = {
            "symbol": symbol,
            "current_price": round(current_price, 2),
            "direction": "Up" if direction == 1 else "Down",
            "confidence": f"{round(confidence, 2)}%",
            "expected_movement": round(movement, 2),
            "predicted_price": round(predicted_price, 2)
        }


        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    


# For Render deployment
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
# For local development
if __name__ == "__main__":
    app.run(debug=True, port=5000)
