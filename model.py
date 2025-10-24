import numpy as np
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers  import *
from tensorflow.keras.layers import LSTM, Dense, InputLayer, Dropout
from tensorflow.keras.losses import MeanSquaredError 
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
import keras_tuner as kt

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

def build_model(hp, X_train):
    model = Sequential()
    model.add(InputLayer((X_train.shape[1], X_train.shape[2])))
    
    lstm_units = hp.Choice('lstm_units', values=[32, 64, 128])
    model.add(LSTM(lstm_units, return_sequences=True))
    
    dropout_rate = hp.Choice('dropout_rate', values=[0.0, 0.2, 0.4])
    model.add(Dropout(dropout_rate))
    
    model.add(LSTM(lstm_units))
    model.add(Dropout(dropout_rate))
    
    activation_function = hp.Choice('activation_function', values=['relu', 'tanh'])
    dense_units = hp.Choice('dense_units', values=[8, 16, 32, 64])  
    model.add(Dense(dense_units, activation=activation_function))
    model.add(Dense(1, activation='linear'))
    
    learning_rate = hp.Choice('learning_rate', values=[0.001, 0.01, 0.1])
    model.compile(
        loss=MeanSquaredError(),
        optimizer=Adam(learning_rate=learning_rate),
        metrics=[RootMeanSquaredError()]
    )
    return model

def build_model_vol1(hp, X_train):
    model = Sequential()
    model.add(InputLayer((X_train.shape[1], X_train.shape[2])))
    
    lstm_units = hp.Choice('lstm_units', values=[32, 64, 128, 256])
    model.add(LSTM(lstm_units, return_sequences=True))
    
    dropout_rate = hp.Choice('dropout_rate', values=[0.0, 0.2, 0.4])
    model.add(Dropout(dropout_rate))
    
    model.add(LSTM(lstm_units))
    model.add(Dropout(dropout_rate))
    
    activation_function = hp.Choice('activation_function', values=['relu', 'tanh'])
    dense_units = hp.Choice('dense_units', values=[16, 32, 64])  
    model.add(Dense(dense_units, activation=activation_function))
    model.add(Dense(1, activation='linear'))
    
    learning_rate = hp.Choice('learning_rate', values=[0.001, 0.01, 0.1])
    model.compile(
        loss=MeanSquaredError(),
        optimizer=Adam(learning_rate=learning_rate),
        metrics=[RootMeanSquaredError()]
    )
    return model