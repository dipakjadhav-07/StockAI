
let stockChart = null;

// ==============================
// GO TO PREDICTION
// ==============================

function goToPrediction() {
    document.getElementById("prediction").scrollIntoView({
        behavior: "smooth"
    });
}


// ==============================
// GENERATE STOCK PREDICTION
// ==============================

async function generatePrediction() {

    const stock = document.getElementById("stock").value;
    const model = document.getElementById("model").value;

    try {
        const response = await fetch(
            `/predict?stock=${encodeURIComponent(stock)}&model=${encodeURIComponent(model)}`
        );

        const data = await response.json();

        if (!response.ok || data.error) {
            throw new Error(data.error || "Prediction failed");
        }

        document.getElementById("result-stock").textContent = stock;
        document.getElementById("result-model").textContent = model;

        document.getElementById("latest-price").textContent =
            Number(data.latest_price).toFixed(2);

        document.getElementById("predicted-price").textContent =
            Number(data.predicted_price).toFixed(2);

        document.getElementById("prediction-direction").textContent =
            data.direction;

        document.getElementById("mae").textContent =
            Number(data.mae).toFixed(2);

        document.getElementById("rmse").textContent =
            Number(data.rmse).toFixed(2);

        document.getElementById("r2").textContent =
            Number(data.r2).toFixed(4);

        document.getElementById("result").style.display = "block";

        // Stock chart
        const chartArea = document.getElementById("stockChart");

        if (stockChart) {
            stockChart.destroy();
        }

        stockChart = new Chart(chartArea, {
            type: "line",

            data: {
                labels: data.dates,

                datasets: [{
                    label: stock + " Historical Price",
                    data: data.prices,
                    borderWidth: 2,
                    tension: 0.3,
                    fill: false
                }]
            },

            options: {
                responsive: true,

                plugins: {
                    legend: {
                        display: true
                    }
                },

                scales: {
                    y: {
                        beginAtZero: false
                    }
                }
            }
        });

    } catch (error) {
        alert("Error: " + error.message);
        console.error(error);
    }
}


// ==============================
// COMPARE ALL ML MODELS
// ==============================

async function compareModels() {

    const stock = document.getElementById("stock").value;

    try {
        const response = await fetch(
            `/compare?stock=${encodeURIComponent(stock)}`
        );

        const data = await response.json();

        if (!response.ok || data.error) {
            throw new Error(data.error || "Model comparison failed");
        }

        const modelIDs = {
            "Linear Regression": "linear",
            "Random Forest": "rf",
            "Gradient Boosting": "gb",
            "Decision Tree": "dt",
            "Extra Trees": "et",
            "AdaBoost": "ada",
            "Support Vector Regression": "svr",
            "KNN": "knn"
        };

        for (const [modelName, prefix] of Object.entries(modelIDs)) {

            const result = data.models[modelName];

            if (!result) continue;

            document.getElementById(`${prefix}-mae`).textContent =
                Number(result.mae).toFixed(2);

            document.getElementById(`${prefix}-rmse`).textContent =
                Number(result.rmse).toFixed(2);

            document.getElementById(`${prefix}-r2`).textContent =
                Number(result.r2).toFixed(4);
        }

        document.getElementById("comparison").scrollIntoView({
            behavior: "smooth"
        });

    } catch (error) {
        alert("Error: " + error.message);
        console.error(error);
    }
}
