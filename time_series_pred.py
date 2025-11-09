import mne
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.preprocessing import OneHotEncoder as ohe
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Input, Flatten, Conv1D, Dropout, Reshape, Bidirectional
from tensorflow.keras.callbacks import ReduceLROnPlateau, Callback
import keras_tuner
from functions import *

######################## definition of variables
p, fname, data, n, chn = init_script("SIGNAL PREDICTION", "to train the model from")
epc = 15     # number of training epochs

stride = n //2
l  = len(chn)

def build_model(p1):
  lstm = Sequential()
  lstm.add(Input( (l,n)))
  lstm.add(Conv1D(filters=p1.Choice('filters', [5, 10, n//2, n, 2*n]),  kernel_size=(l)))
  lstm.add(Bidirectional(LSTM(units=(p1.Choice('neurons 1', [n, 2*n, 5*n]) ), return_sequences=False)))
  lstm.add(Dropout(rate=p1.Choice("dropout rate", [0.2,0.3, 0.4, 0.45])))

  lstm.add(Dense(n*l))
  lstm.add(Reshape((l, n)))

  lstm.compile(loss='mse', metrics=['mean_absolute_error'], optimizer='adam')
  return lstm

def test_model(x, y, idx):
    pred = lstm.predict(x[idx].reshape(1,l, n))
    real = y[idx]

    plt.plot(pred[0][0], label='pred')
    plt.plot(real[0], label='real')
    plt.legend()
    plt.show()

    return pred, real

x, _ = split_seq(data,chn, n, stride, True)

# the last window is not included in the 'x' (training input) set,
# while the 'y' set (what is predicted) does not include the first window

y = x[1:]
x = x[:-1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
# model training
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.0001)
tuner = keras_tuner.RandomSearch(build_model, objective='val_loss')
tuner.search(x_train, y_train, epochs=epc, validation_data=(x_test, y_test), callbacks=[reduce_lr])
lstm = tuner.get_best_models()[0]

print("\n\nEnd of training\nModel evaluation: ")
print("Best hyperperparameters discovered:\n  ", tuner.get_best_hyperparameters()[0].values, "\n")

pred, real = test_model(x_test,y_test, 10)

print("\nSome predicted values from the first channel: ", pred[0, 0,:6])
print("Some real values from from the first channel  :      ", real[0,:6])

sv_model(lstm)
