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
// GENERATE PREDICTION
// ==============================

async function generatePrediction() {

    const stock =
        document.getElementById("stock").value;

    const model =
        document.getElementById("model").value;


    try {

        const response = await fetch(
            `/predict?stock=${encodeURIComponent(stock)}&model=${encodeURIComponent(model)}`
        );


        const data = await response.json();


        if (data.error) {

            alert(data.error);

            return;

        }


        // ==============================
        // PREDICTION INFORMATION
        // ==============================

        document.getElementById("result-stock").textContent =
            data.stock;


        document.getElementById("result-model").textContent =
            data.model;


        document.getElementById("latest-price").textContent =
            "₹" + data.latest_price;


        document.getElementById("predicted-price").textContent =
            "₹" + data.predicted_price;


        document.getElementById("prediction-direction").textContent =
            data.direction;


        // ==============================
        // MODEL PERFORMANCE
        // ==============================

        document.getElementById("mae").textContent =
            data.mae;


        document.getElementById("rmse").textContent =
            data.rmse;


        document.getElementById("r2").textContent =
            data.r2;


        // ==============================
        // SHOW RESULT
        // ==============================

        document.getElementById("result").style.display =
            "block";


        // ==============================
        // HISTORICAL CHART
        // ==============================

        const chartArea =
            document.getElementById("stockChart");


        // Remove old chart

        if (stockChart !== null) {

            stockChart.destroy();

        }


        // Create new chart

        stockChart = new Chart(chartArea, {

            type: "line",

            data: {

                labels: data.dates,

                datasets: [{

                    label:
                        data.stock + " Closing Price",

                    data:
                        data.prices,

                    borderWidth: 2,

                    fill: false,

                    tension: 0.3

                }]

            },


            options: {

                responsive: true,

                maintainAspectRatio: false,

                plugins: {

                    legend: {

                        display: true

                    }

                },


                scales: {

                    x: {

                        title: {

                            display: true,

                            text: "Date"

                        }

                    },


                    y: {

                        title: {

                            display: true,

                            text: "Price (₹)"

                        }

                    }

                }

            }

        });

    }


    catch (error) {

        console.error(error);

        alert(
            "Unable to connect to the prediction server."
        );

    }

}



// ==============================
// COMPARE MODELS
// ==============================


async function compareModels() {
    const stock = document.getElementById("stock").value;

    try {
        const response = await fetch(
            `/compare?stock=${encodeURIComponent(stock)}`
        );

        const data = await response.json();

        if (!response.ok || data.error) {
            alert(data.error || "Unable to compare models.");
            return;
        }

        const modelRows = {
            "Linear Regression": "linear",
            "Random Forest": "rf",
            "Gradient Boosting": "gb",
            "Decision Tree": "dt",
            "Extra Trees": "et",
            "AdaBoost": "ada",
            "Support Vector Regression": "svr",
            "KNN": "knn"
        };

        for (const [modelName, prefix] of Object.entries(modelRows)) {
            const result = data.models[modelName];

            if (!result) continue;

            const mae = document.getElementById(`${prefix}-mae`);
            const rmse = document.getElementById(`${prefix}-rmse`);
            const r2 = document.getElementById(`${prefix}-r2`);

            if (mae) mae.textContent = result.mae;
            if (rmse) rmse.textContent = result.rmse;
            if (r2) r2.textContent = result.r2;
        }
    } catch (error) {
        console.error("Comparison error:", error);
        alert("Unable to compare models.");
    }
}


        // ==============================
        // LINEAR REGRESSION
        // ==============================

        document.getElementById("linear-mae").textContent =
            data.linear_regression.mae;


        document.getElementById("linear-rmse").textContent =
            data.linear_regression.rmse;


        document.getElementById("linear-r2").textContent =
            data.linear_regression.r2;



        // ==============================
        // RANDOM FOREST
        // ==============================

        document.getElementById("rf-mae").textContent =
            data.random_forest.mae;


        document.getElementById("rf-rmse").textContent =
            data.random_forest.rmse;


        document.getElementById("rf-r2").textContent =
            data.random_forest.r2;


        // Success message

        

    }


    catch (error) {

        console.error(
            "Compare error:",
            error
        );


        alert(
            "Unable to compare the models."
        );

    }

}