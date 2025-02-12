import pandas as pd
import numpy as np
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose


df = pd.read_csv("dataset/processed/train/train.csv", parse_dates=['time'], index_col='time')

result = seasonal_decompose(df['price actual'], model='additative', period=12*24)

trend = result.trend
seasonal = result.seasonal
residual = result.resid

fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)

df['price actual'][:168*3].plot(ax=axes[0], title='Original')
axes[0].set_ylabel('Price')

trend[:168*3].plot(ax=axes[1], title='Trend')
axes[1].set_ylabel('Trend')

seasonal[:168*3].plot(ax=axes[2], title='Seasonal')
axes[2].set_ylabel('Seasonal')

residual[:168*3].plot(ax=axes[3], title='Residual')
axes[3].set_ylabel('Residual')

plt.tight_layout()
plt.show()
