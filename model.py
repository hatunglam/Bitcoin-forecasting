import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers  import *
from tensorflow.keras.losses import MeanSquaredError 
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam

def close_price_model(df, wind_size=30):
    df_np = df
    X = [] 
    y = []
    for i in range(len(df_np) - wind_size):
        row = [a for a in df_np[i : i+wind_size]] 
        X.append(row)
        label = df_np[i+wind_size][3] # target index
        y.append(label)
    return np.array(X), np.array(y) 

def vanilla_model(X_train):
    model = Sequential()
    model.add(InputLayer((X_train.shape[1], X_train.shape[2])))
    model.add(LSTM(64))
    model.add(Dense(8, 'relu'))
    model.add(Dense(1, 'linear'))

    model.summary()

    return model