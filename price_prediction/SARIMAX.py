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


df = pd.read_csv(
    "dataset/processed/train/train.csv", parse_dates=["time"], index_col="time"
)

#take the first year of the data
df = df[: 24 * 30]

result = seasonal_decompose(df["price actual"], model="additative", period=24)

trend = result.trend
seasonal = result.seasonal
residual = result.resid

fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)

df["price actual"][: 168 * 3].plot(ax=axes[0], title="Original")
axes[0].set_ylabel("Price")

trend[: 168 * 3].plot(ax=axes[1], title="Trend")
axes[1].set_ylabel("Trend")

seasonal[: 168 * 3].plot(ax=axes[2], title="Seasonal")
axes[2].set_ylabel("Seasonal")

residual[: 168 * 3].plot(ax=axes[3], title="Residual")
axes[3].set_ylabel("Residual")

plt.tight_layout()
plt.show()



SARIMAX_model = pm.auto_arima(
    df["price actual"],
    start_p=1,
    start_q=1,
    test="adf",
    max_p=3,
    max_q=3,
    m=12,
    start_P=0,
    seasonal=True,
    d=None,
    D=1,
    trace=False,
    error_action="ignore",
    suppress_warnings=True,
    stepwise=True,
)

print(SARIMAX_model.summary())