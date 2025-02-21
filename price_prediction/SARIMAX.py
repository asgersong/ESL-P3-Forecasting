from importlib import simple
import pandas as pd
import numpy as np
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
import pmdarima as pm
from tqdm import tqdm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

df = pd.read_csv(
    "dataset/processed/train/train.csv", parse_dates=["time"], index_col="time"
)

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

exog_train = df[["windpower", "solarpower"]].shift(24)
exog_train = exog_train[24:]
df = df[24:]

print(exog_train.describe(include="all"))
print(df["price actual"].describe(include="all"))
print(exog_train.head())
print(df["price actual"].head())

# plot_acf(df["price actual"], lags=365)
# plt.show()
# plot_pacf(df["price actual"], lags=365)
# plt.show()

# Initialize variables
n_periods = 24
train_size = 200
test_size = 120
predictions = []
predictions_var = []

# Fit the initial model
SARIMAX_model = SARIMAX(
    df["price actual"][:train_size],
    exog=exog_train[:train_size],
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 24),
    simple_differencing=False,
)
# Initialize progress bar
progress = tqdm(total=100, desc="Training Auto-ARIMA", bar_format="{l_bar}{bar} [ time left: {remaining} ]")

def update_progress(*args, **kwargs):
    progress.update(2)

SARIMAX_result = SARIMAX_model.fit(disp=True, callback=update_progress)
progress.close()

# Loop to predict and update the model
for i in tqdm(range(0, test_size, n_periods), desc="Forecasting", bar_format="{l_bar}{bar} [ time left: {remaining} ]"):
    
    # Forecast
    forecast = SARIMAX_result.get_forecast(
        steps=n_periods, exog=exog_train[train_size + i : train_size + i + n_periods]
    )
    predictions.extend(forecast.predicted_mean)
    predictions_var.extend(forecast.var_pred_mean)

    # Update the model with new data
    SARIMAX_result = SARIMAX_result.append(
        df["price actual"][train_size + i : train_size + i + n_periods],
        exog=exog_train[train_size + i : train_size + i + n_periods],
        refit=True,
    )

# Plot the results
plt.plot(df.index[:train_size], df["price actual"][:train_size], label="Train")
plt.plot(df.index[train_size : train_size + test_size], predictions, label="Forecast")
plt.fill_between(
    df.index[train_size : train_size + test_size],
    predictions - np.sqrt(predictions_var),
    predictions + np.sqrt(predictions_var),
    alpha=0.2,
)
plt.plot(
    df.index[train_size : train_size + test_size],
    df["price actual"][train_size : train_size + test_size],
    label="Actual",
)
plt.legend()
plt.show()
print(SARIMAX_result.summary())
