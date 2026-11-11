
from flask import Flask, jsonify, send_from_directory, request
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
    AdaBoostRegressor
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import numpy as np

app = Flask(__name__)

stock_symbols = {
    "TCS": "TCS.NS",
    "Reliance Industries": "RELIANCE.NS",
    "Infosys": "INFY.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "State Bank of India": "SBIN.NS",
    "ITC": "ITC.NS",
    "Wipro": "WIPRO.NS"
}


# HOME PAGE
@app.route("/")
def home():
    return send_from_directory(".", "index.html")


# CSS AND JAVASCRIPT
@app.route("/style.css")
def css():
    return send_from_directory(".", "style.css")


@app.route("/script.js")
def javascript():
    return send_from_directory(".", "script.js")


# GET STOCK DATA
def get_clean_stock_data(selected_stock):

    if selected_stock not in stock_symbols:
        raise ValueError("Invalid stock selected")

    stock = yf.download(
        stock_symbols[selected_stock],
        period="2y",
        auto_adjust=False,
        progress=False
    )

    if stock.empty:
        raise ValueError("Unable to download stock data")

    close_price = stock["Close"]

    if isinstance(close_price, pd.DataFrame):
        close_price = close_price.iloc[:, 0]

    close_price = pd.to_numeric(
        close_price, errors="coerce"
    )

    close_price = close_price.replace(
        [np.inf, -np.inf], np.nan
    ).dropna()

    if len(close_price) < 20:
        raise ValueError("Not enough stock data")

    return close_price


# MODEL SELECTION
def get_model(name):

    models = {
        "Linear Regression": LinearRegression(),

        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            random_state=42
        ),

        "Gradient Boosting": GradientBoostingRegressor(
            random_state=42
        ),

        "Decision Tree": DecisionTreeRegressor(
            random_state=42
        ),

        "Extra Trees": ExtraTreesRegressor(
            n_estimators=100,
            random_state=42
        ),

        "AdaBoost": AdaBoostRegressor(
            random_state=42
        ),

        "Support Vector Regression": make_pipeline(
            StandardScaler(), SVR()
        ),

        "KNN": make_pipeline(
            StandardScaler(),
            KNeighborsRegressor(n_neighbors=5)
        )
    }

    return models.get(name, LinearRegression())


# CREATE DATASET
def prepare_data(close_price):

    data = pd.DataFrame({
        "Close": close_price.astype(float)
    })

    data["Previous_Close"] = data["Close"].shift(1)

    return data.dropna()


# PREDICTION
@app.route("/predict")
def predict():

    try:
        selected_stock = request.args.get("stock", "TCS")
        selected_model = request.args.get(
            "model", "Linear Regression"
        )

        close_price = get_clean_stock_data(selected_stock)
        data = prepare_data(close_price)

        X = data[["Previous_Close"]]
        y = data["Close"]

        split = int(len(data) * 0.80)

        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        if len(X_test) == 0:
            raise ValueError("Not enough data for testing")

        model = get_model(selected_model)

        model.fit(X_train, y_train)
        test_predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, test_predictions)

        rmse = np.sqrt(
            mean_squared_error(y_test, test_predictions)
        )

        r2 = r2_score(y_test, test_predictions)

        # Train final model on all data
        model.fit(X, y)

        latest_close = float(close_price.iloc[-1])

        latest_data = pd.DataFrame({
            "Previous_Close": [latest_close]
        })

        prediction = float(model.predict(latest_data)[0])

        if prediction > latest_close:
            direction = "Expected to Rise ↑"
        elif prediction < latest_close:
            direction = "Expected to Fall ↓"
        else:
            direction = "No Change →"

        chart_data = data.tail(30)

        dates = [
            date.strftime("%Y-%m-%d")
            for date in chart_data.index
        ]

        prices = [
            round(float(price), 2)
            for price in chart_data["Close"]
        ]

        return jsonify({
            "stock": selected_stock,
            "model": selected_model,
            "latest_price": round(latest_close, 2),
            "predicted_price": round(prediction, 2),
            "direction": direction,
            "mae": round(float(mae), 2),
            "rmse": round(float(rmse), 2),
            "r2": round(float(r2), 4),
            "dates": dates,
            "prices": prices
        })

    except Exception as e:
        print("Prediction Error:", str(e))
        return jsonify({"error": str(e)}), 500


# COMPARE ALL MODELS
@app.route("/compare")
def compare():

    try:
        selected_stock = request.args.get("stock", "TCS")

        close_price = get_clean_stock_data(selected_stock)
        data = prepare_data(close_price)

        X = data[["Previous_Close"]]
        y = data["Close"]

        split = int(len(data) * 0.80)

        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        if len(X_test) == 0:
            raise ValueError("Not enough data for comparison")

        model_names = [
            "Linear Regression",
            "Random Forest",
            "Gradient Boosting",
            "Decision Tree",
            "Extra Trees",
            "AdaBoost",
            "Support Vector Regression",
            "KNN"
        ]

        results = {}

        for name in model_names:

            model = get_model(name)

            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            results[name] = {
                "mae": round(float(
                    mean_absolute_error(y_test, predictions)
                ), 2),

                "rmse": round(float(
                    np.sqrt(mean_squared_error(y_test, predictions))
                ), 2),

                "r2": round(float(
                    r2_score(y_test, predictions)
                ), 4)
            }

        return jsonify({
            "stock": selected_stock,
            "models": results
        })

    except Exception as e:
        print("Comparison Error:", str(e))
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=False)
