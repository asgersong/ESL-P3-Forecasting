import pandas as pd
import numpy as np
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose

df_y = pd.read_csv(r"dataset\processed\train\train_y.csv")

<<<<<<< Updated upstream
# df = pd.read_csv(r"dataset\processed\train\train_x.csv")

df = pd.read_csv("../dataset/processed/train/train_x.csv")
=======
result = seasonal_decompose(df_y)
>>>>>>> Stashed changes
