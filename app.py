from flask import Flask, jsonify, render_template, send_from_directory, request
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


# =========================================================
# HOME PAGE
# =========================================================

@app.route("/")
def home():
    return render_template("index.html")


# =========================================================
# SERVE CSS AND JAVASCRIPT
# =========================================================

@app.route("/<path:filename>")
def serve_file(filename):
    return send_from_directory(".", filename)


# =========================================================
# STOCK SYMBOLS
# =========================================================


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


# =========================================================
# GET CLEAN STOCK DATA
# =========================================================

def get_clean_stock_data(selected_stock):

    if selected_stock not in stock_symbols:
        raise ValueError("Invalid stock selected.")

    symbol = stock_symbols[selected_stock]

    stock = yf.download(
        symbol,
        period="2y",
        auto_adjust=False,
        progress=False
    )

    if stock.empty:
        raise ValueError("Unable to download stock data.")

    # Get Close price
    close_price = stock["Close"]

    # Yahoo Finance may return a DataFrame
    if isinstance(close_price, pd.DataFrame):
        close_price = close_price.iloc[:, 0]

    # Convert all values to numeric
    close_price = pd.to_numeric(
        close_price,
        errors="coerce"
    )

    # Remove infinity values
    close_price = close_price.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Remove NaN values
    close_price = close_price.dropna()

    if close_price.empty:
        raise ValueError("No valid closing price data found.")

    return close_price


# =========================================================
# PREDICT
# =========================================================

@app.route("/predict")
def predict():

    try:

        # Get selected stock and model
        selected_stock = request.args.get(
            "stock",
            "TCS"
        )

        selected_model = request.args.get(
            "model",
            "Linear Regression"
        )


        # Get clean data
        close_price = get_clean_stock_data(
            selected_stock
        )


        # =================================================
        # CREATE DATASET
        # =================================================

        data = pd.DataFrame()

        data["Close"] = close_price.astype(float)

        data["Previous_Close"] = (
            data["Close"].shift(1)
        )

        data = data.dropna()


        # Input and target
        X = data[["Previous_Close"]]
        y = data["Close"]


        # =================================================
        # SPLIT DATA
        # =================================================

        split = int(len(data) * 0.80)

        X_train = X.iloc[:split]
        X_test = X.iloc[split:]

        y_train = y.iloc[:split]
        y_test = y.iloc[split:]


        # =================================================
        # SELECT MODEL
        # =================================================

        if selected_model == "Random Forest":
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

elif selected_model == "Gradient Boosting":
    model = GradientBoostingRegressor(random_state=42)

elif selected_model == "Decision Tree":
    model = DecisionTreeRegressor(random_state=42)

elif selected_model == "Extra Trees":
    model = ExtraTreesRegressor(
        n_estimators=100,
        random_state=42
    )

elif selected_model == "AdaBoost":
    model = AdaBoostRegressor(random_state=42)

elif selected_model == "Support Vector Regression":
    model = make_pipeline(
        StandardScaler(),
        SVR()
    )

elif selected_model == "KNN":
    model = make_pipeline(
        StandardScaler(),
        KNeighborsRegressor(n_neighbors=5)
    )

else:
    model = LinearRegression()


        # =================================================
        # TRAIN MODEL
        # =================================================

        model.fit(
            X_train,
            y_train
        )


        # =================================================
        # TEST PREDICTION
        # =================================================

        test_predictions = model.predict(
            X_test
        )


        # =================================================
        # PERFORMANCE
        # =================================================

        mae = mean_absolute_error(
            y_test,
            test_predictions
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                test_predictions
            )
        )

        r2 = r2_score(
            y_test,
            test_predictions
        )


        # =================================================
        # TRAIN FINAL MODEL USING ALL DATA
        # =================================================

        model.fit(
            X,
            y
        )


        # =================================================
        # LATEST PRICE
        # =================================================

        latest_close = float(
            close_price.iloc[-1]
        )


        # =================================================
        # NEXT PRICE PREDICTION
        # =================================================

        latest_data = pd.DataFrame({
            "Previous_Close": [
                latest_close
            ]
        })


        prediction_array = model.predict(
            latest_data
        )


        prediction = float(
            prediction_array[0]
        )


        # =================================================
        # DIRECTION
        # =================================================

        if prediction > latest_close:

            direction = "Expected to Rise ↑"

        elif prediction < latest_close:

            direction = "Expected to Fall ↓"

        else:

            direction = "No Change →"


        # =================================================
        # CHART DATA
        # =================================================

        chart_data = data.tail(30)


        chart_dates = [
            date.strftime("%Y-%m-%d")
            for date in chart_data.index
        ]


        chart_prices = [
            round(
                float(price),
                2
            )
            for price in chart_data["Close"]
        ]


        # =================================================
        # SEND RESULT
        # =================================================

        return jsonify({

            "stock": selected_stock,

            "model": selected_model,

            "latest_price": round(
                latest_close,
                2
            ),

            "predicted_price": round(
                prediction,
                2
            ),

            "direction": direction,

            "mae": round(
                float(mae),
                2
            ),

            "rmse": round(
                float(rmse),
                2
            ),

            "r2": round(
                float(r2),
                4
            ),

            "dates": chart_dates,

            "prices": chart_prices

        })


    except Exception as e:

        print(
            "Prediction Error:",
            str(e)
        )

        return jsonify({
            "error": str(e)
        }), 500


# =========================================================
# COMPARE MODELS
# =========================================================
@app.route("/compare")
def compare():
    try:
        selected_stock = request.args.get("stock", "TCS")
        close_price = get_clean_stock_data(selected_stock)

        data = pd.DataFrame()
        data["Close"] = close_price.astype(float)
        data["Previous_Close"] = data["Close"].shift(1)
        data = data.dropna()

        X = data[["Previous_Close"]]
        y = data["Close"]

        split = int(len(data) * 0.80)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(
                n_estimators=100, random_state=42
            ),
            "Gradient Boosting": GradientBoostingRegressor(
                random_state=42
            ),
            "Decision Tree": DecisionTreeRegressor(
                random_state=42
            ),
            "Extra Trees": ExtraTreesRegressor(
                n_estimators=100, random_state=42
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

        results = {}

        for name, model in models.items():
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            results[name] = {
                "mae": round(float(mean_absolute_error(
                    y_test, predictions
                )), 2),
                "rmse": round(float(np.sqrt(mean_squared_error(
                    y_test, predictions
                ))), 2),
                "r2": round(float(r2_score(
                    y_test, predictions
                )), 4)
            }

        return jsonify({
            "stock": selected_stock,
            "models": results
        })

    except Exception as e:
        print("Comparison Error:", str(e))
        return jsonify({"error": str(e)}), 500

    try:

        selected_stock = request.args.get(
            "stock",
            "TCS"
        )


        # Get clean data
        close_price = get_clean_stock_data(
            selected_stock
        )


        # =================================================
        # CREATE DATASET
        # =================================================

        data = pd.DataFrame()

        data["Close"] = close_price.astype(float)

        data["Previous_Close"] = (
            data["Close"].shift(1)
        )

        data = data.dropna()


        X = data[["Previous_Close"]]
        y = data["Close"]


        # =================================================
        # SPLIT DATA
        # =================================================

        split = int(len(data) * 0.80)


        X_train = X.iloc[:split]
        X_test = X.iloc[split:]

        y_train = y.iloc[:split]
        y_test = y.iloc[split:]


        # =================================================
        # LINEAR REGRESSION
        # =================================================

        linear_model = LinearRegression()


        linear_model.fit(
            X_train,
            y_train
        )


        linear_prediction = linear_model.predict(
            X_test
        )


        linear_mae = mean_absolute_error(
            y_test,
            linear_prediction
        )


        linear_rmse = np.sqrt(
            mean_squared_error(
                y_test,
                linear_prediction
            )
        )


        linear_r2 = r2_score(
            y_test,
            linear_prediction
        )


        # =================================================
        # RANDOM FOREST
        # =================================================

        rf_model = RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )


        rf_model.fit(
            X_train,
            y_train
        )


        rf_prediction = rf_model.predict(
            X_test
        )


        rf_mae = mean_absolute_error(
            y_test,
            rf_prediction
        )


        rf_rmse = np.sqrt(
            mean_squared_error(
                y_test,
                rf_prediction
            )
        )


        rf_r2 = r2_score(
            y_test,
            rf_prediction
        )


        # =================================================
        # RETURN COMPARISON
        # =================================================

        return jsonify({

            "stock": selected_stock,

            "linear_regression": {

                "mae": round(
                    float(linear_mae),
                    2
                ),

                "rmse": round(
                    float(linear_rmse),
                    2
                ),

                "r2": round(
                    float(linear_r2),
                    4
                )

            },

            "random_forest": {

                "mae": round(
                    float(rf_mae),
                    2
                ),

                "rmse": round(
                    float(rf_rmse),
                    2
                ),

                "r2": round(
                    float(rf_r2),
                    4
                )

            }

        })


    except Exception as e:

        print(
            "Comparison Error:",
            str(e)
        )

        return jsonify({
            "error": str(e)
        }), 500


# =========================================================
# RUN FLASK
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )