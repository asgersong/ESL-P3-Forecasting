import json
import os
from datetime import datetime
from importlib import simple
from re import S

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import pmdarima as pm
from matplotlib.pylab import rcParams
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.sarimax import SARIMAXResults
from statsmodels.tsa.stattools import adfuller
from tqdm import tqdm

df_train = pd.read_csv(
    "dataset/processed/train/train.csv", parse_dates=["time"], index_col="time"
)

df_test = pd.read_csv(
    "dataset/processed/test/test.csv", parse_dates=["time"], index_col="time"
)

df = pd.concat([df_train, df_test], axis=0)

# take the first year of the data
# print(df.describe(include="all"))
# result = seasonal_decompose(df["price actual"], model="additative", period=24)

# trend = result.trend
# seasonal = result.seasonal
# residual = result.resid

# fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)

# df["price actual"][: 168 * 3].plot(ax=axes[0], title="Original")
# axes[0].set_ylabel("Price")

# trend[: 168 * 3].plot(ax=axes[1], title="Trend")
# axes[1].set_ylabel("Trend")

# seasonal[: 168 * 3].plot(ax=axes[2], title="Seasonal")
# axes[2].set_ylabel("Seasonal")

# residual[: 168 * 3].plot(ax=axes[3], title="Residual")
# axes[3].set_ylabel("Residual")

# plt.tight_layout()
# plt.show()

exog_train = df[
    [
        "total_load_actual",
        "hour_of_day",
        "windpower",
        "solarpower",
        "fossil_fuels",
        "other_green_energy",
    ]
].shift(24)
exog_train["day_of_week"] = df.index.dayofweek
exog_train = exog_train[24:]
df = df[24:]

print(exog_train.describe(include="all"))
print(df["price_actual"].describe(include="all"))
print(exog_train.head())
print(df["price_actual"].head())

# plot_acf(df["price actual"], lags=365)
# plt.show()
# plot_pacf(df["price actual"], lags=365)
# plt.show()

# Initialize variables
period = 24 * 14
train_size = 24 * 100
horizon = 24
predictions = []
predictions_var = []
predictions_index = []
logs = {}
model_name = "SARIMAX_model.pkl"

progress = tqdm(
    total=100,
    desc="Training Auto-ARIMA",
    bar_format="{l_bar}{bar} [ time left: {remaining} ]",
)


def update_progress(*args, **kwargs):
    progress.update(2)


df = df[: train_size + period + horizon]
# load model if exists
if os.path.exists(f"price_prediction/SARIMAX/models/{model_name}"):
    SARIMAX_result = SARIMAXResults.load(
        f"price_prediction/SARIMAX/models/{model_name}"
    )
    SARIMAX_model = SARIMAX_result.model
else:
    # Fit the initial model
    SARIMAX_model = SARIMAX(
        df["price_actual"][:train_size],
        exog=exog_train[:train_size],
        order=(0, 0, 1),
        seasonal_order=(1, 1, 1, 24),
        simple_differencing=False,
    )

    # Automatically find optimal SARIMAX parameters
    # model = pm.auto_arima(
    #     df["price_actual"][:train_size],
    #     exogenous=exog_train[:train_size],
    #     seasonal=True,
    #     m=24,  # Seasonal period
    #     trace=True,
    #     error_action="ignore",
    #     suppress_warnings=True,
    #     stepwise=True,
    # )

    # print(model.summary())

    # # Refit the model with optimal parameters
    # SARIMAX_model = SARIMAX(
    #     df["price_actual"][:train_size],
    #     exog=exog_train[:train_size],
    #     order=model.order,
    #     seasonal_order=model.seasonal_order,
    # )
    # SARIMAX_result = SARIMAX_model.fit()

    # Initialize progress bar

    SARIMAX_result = SARIMAX_model.fit(disp=True, callback=update_progress)
    progress.close()

    # save the model
    os.makedirs("models", exist_ok=True)
    SARIMAX_result.save(f"price_prediction/SARIMAX/models/{model_name}")

print(SARIMAX_result.summary())
logs["order"] = SARIMAX_model.order
logs["seasonal_order"] = SARIMAX_model.seasonal_order
logs["summary"] = SARIMAX_result.summary().as_text()


# Loop to predict and update the model
for i in tqdm(
    range(0, len(df) - train_size, period),
    desc="Forecasting",
    bar_format="{l_bar}{bar} [ time left: {remaining} ]",
):
    # Forecast
    forecast = SARIMAX_result.get_forecast(
        steps=horizon, exog=exog_train[train_size + i : train_size + i + horizon]
    )
    predictions.extend(forecast.predicted_mean)
    predictions_var.extend(forecast.var_pred_mean)
    predictions_index.extend(df.index[train_size + i : train_size + i + horizon])

    # Calculate MAE and RMSE
    actual = df["price_actual"].loc[predictions_index]
    mae = mean_absolute_error(actual[-horizon:], predictions[-horizon:])
    rmse = np.sqrt(mean_squared_error(actual[-horizon:], predictions[-horizon:]))

    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")

    logs[f"MAE_{i}"] = mae
    logs[f"RMSE_{i}"] = rmse

    # Update the model with new data
    # SARIMAX_result = SARIMAX_result.append(
    #     df["price_actual"][train_size + i : train_size + i + n_periods],
    #     exog=exog_train[train_size + i : train_size + i + n_periods],
    #     refit=False,
    # )

    if i + period >= len(df) - train_size:
        break
    SARIMAX_model = SARIMAX(
        df["price_actual"][: train_size + period + i],
        exog=exog_train[: train_size + period + i],
        order=SARIMAX_model.order,
        seasonal_order=SARIMAX_model.seasonal_order,
        simple_differencing=False,
    )
    SARIMAX_result = SARIMAX_model.fit(disp=True, callback=update_progress)


# MAE and RMSE
actual = df["price_actual"].loc[predictions_index]
mae = mean_absolute_error(actual, predictions)
rmse = np.sqrt(mean_squared_error(actual, predictions))

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

logs["MAE"] = mae
logs["RMSE"] = rmse

# Plot the results
plt.plot(
    df.index,
    df["price_actual"],
    label="Actual",
)
plt.legend()
for i in range(0, len(predictions), horizon):
    plt.plot(
        predictions_index[i : i + horizon],
        predictions[i : i + horizon],
        label="Forecast",
    )
    plt.fill_between(
        predictions_index[i : i + horizon],
        predictions[i : i + horizon] - np.sqrt(predictions_var[i : i + horizon]),
        predictions[i : i + horizon] + np.sqrt(predictions_var[i : i + horizon]),
        alpha=0.2,
    )
plt.show()
print(SARIMAX_result.summary())

# Save the logs
logs = json.dumps(logs)
with open("price_prediction/SARIMAX/logs/SARIMAX_logs.json", "w") as f:
    f.write(logs)
