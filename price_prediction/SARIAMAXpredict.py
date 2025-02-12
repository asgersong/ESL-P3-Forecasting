import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima import auto_arima

# Load dataset
df = pd.read_csv("dataset/processed/train/train.csv", parse_dates=['time'], index_col='time')

# Ensure data is sorted
df = df.sort_index()

# Take the first year of the data
df = df[:24*60]

# Perform seasonal decomposition
result = seasonal_decompose(df['price actual'], model='additive', period=12*24)

# Extract components
trend = result.trend
seasonal = result.seasonal
residual = result.resid

# Check stationarity
adf_test = adfuller(df['price actual'].dropna())
print(f"ADF Statistic: {adf_test[0]}")
print(f"p-value: {adf_test[1]}")

# Differencing if needed
if adf_test[1] > 0.05:
    df['price_diff'] = df['price actual'].diff().dropna()
    target_col = 'price_diff'
else:
    target_col = 'price actual'

# Train-Test Split
train_size = int(len(df) * 0.8)
train, test = df[:train_size], df[train_size:]

# Start timer
start_time = time.time()

# Initialize progress bar
progress = tqdm(total=100, desc="Training Auto-ARIMA", bar_format="{l_bar}{bar} [ time left: {remaining} ]")

def update_progress(*args, **kwargs):
    progress.update(10)

# Train Auto-ARIMA model
model = auto_arima(train[target_col], seasonal=True, m=24, trace=True, stepwise=True, callback=update_progress)
progress.close()

# End timer
end_time = time.time()
training_time = end_time - start_time
print(f"Training completed in {training_time:.2f} seconds")

# Forecast
forecast = model.predict(n_periods=len(test))

# Plot results
plt.figure(figsize=(12,6))
plt.plot(train.index, train['price actual'], label='Train')
plt.plot(test.index, test['price actual'], label='Test', color='orange')
plt.plot(test.index, forecast, label='Predicted', color='red')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Price Forecasting using Auto-ARIMA')
plt.show()
