from flask import Flask, request, jsonify
from flask_cors import CORS
from joblib import load
import yfinance as yf
import pandas as pd

app = Flask(__name__)
CORS(app)

clf = load("model_direction.pkl")
reg = load("model_movement.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    symbol = data.get("symbol").strip().upper()

    try:
        df = yf.download(symbol, period="7d", interval="1d")
        df.dropna(inplace=True)
        X = df[['Open', 'High', 'Low', 'Close', 'Volume']].iloc[-1:]

        direction = clf.predict(X)[0]
        movement = reg.predict(X)[0]

        result = {
            "symbol": symbol.upper(),
            "direction": "Up" if direction == 1 else "Down",
            "movement": round(abs(movement), 2)
        }

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
    