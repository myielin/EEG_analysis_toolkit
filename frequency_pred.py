import mne
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.preprocessing import OneHotEncoder as ohe
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Input, Flatten, Conv1D, Dropout, Reshape
from tensorflow.keras.callbacks import ReduceLROnPlateau, Callback
import keras_tuner
from functions import *

######################## definition of variables
p, fname, data, n, chn = init_script("FREQUENCY BAND PREDICTION", "to train the model from")

epc=30
stride = int(n//2)                             # stride for the sliding window
l = len(chn)            # an important value that is used for dimentioning

def build_model(p1):
  lstm = Sequential()

  lstm.add(Input( (l,n)))
  lstm.add(Conv1D(filters=p1.Choice('filters', [5, 10, n//2, n, 2*n]),  kernel_size=(l)))
  lstm.add(LSTM(units=(p1.Choice('neurons 1', [n, 2*n, 5*n]) ), return_sequences=True))
  lstm.add(Dropout(rate=p1.Choice("dropout rate", [0.2, 0.3, 0.4, 0.45])))
  lstm.add(LSTM(units=(p1.Choice('neurons 2', [n, 2*n, 5*n]) ), return_sequences=False))
  lstm.add(Dense(outShape, activation="softmax"))

  lstm.compile(loss='categorical_crossentropy', metrics=['mean_absolute_error', 'accuracy'], optimizer='adam')

  return lstm

def test_model(x, y, idx):
  preds, reals = [], []
  for i in range(idx, idx+10):
    pval = lstm.predict(x[i].reshape(1,l, n))
    preds.append(int(vals[np.argmax(pval)]))
    reals.append(int(vals[np.argmax(y[i])]))


  print(f"Some predicted values starting at index {idx}::\n{preds}")
  print(f"Real values starting at index {idx}          ::\n{reals}")

  return preds, reals

x, y = split_seq(data, chn, n, stride, normalize=False)

# the last window is not included in the 'x' (training input) set,
# while the 'y' set (what is predicted) does not include the first window
# in this context it is also necessary to one-hot encode the y variable

y = y[1:]
o = ohe( sparse_output=False)
y_ohe = o.fit_transform(y.reshape(-1,1))
vals = np.unique(y)
outShape = len(vals)
x = x[:-1]
x_train, x_test, y_train, y_test = train_test_split(x, y_ohe, test_size=0.2)
# model training
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.2, patience=2, min_lr=0.0001)
tuner = keras_tuner.RandomSearch(build_model, objective='val_accuracy')
tuner.search(x_train, y_train, epochs=epc, validation_data=(x_test, y_test), callbacks=[reduce_lr])
lstm = tuner.get_best_models()[0]

print("\n\nEnd of training\nModel evaluation: ")
print("Best hyperperparameters discovered:\n  ", tuner.get_best_hyperparameters()[0].values, "\n")

pred, real = test_model(x_test,y_test, 14)


sv_model(lstm)
