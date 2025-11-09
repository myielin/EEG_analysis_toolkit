from mne_realtime import MockRtClient,  RtEpochs
from mne_lsl.player import PlayerLSL as Player
from mne_lsl.stream import StreamLSL as Stream
from mne_lsl.stream import EpochsStream as eStream
import mne
import warnings
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tensorflow.keras.saving import load_model
import time
import numpy as np
from functions import init_script, split_seq, norm
warnings.filterwarnings("ignore")

p, fname, data, n, ch = init_script("REAL-TIME PREDICTION", "to simulate real-time data acquisition from")
mn =  input("Enter the trained model name without the .keras extension:  ")
model = load_model(p+"models/"+mn+".keras")
_, freqs = split_seq(data, ch, n, n//2, True) # fetch all frequencies from the data that the model was trained on

rt = Player(data, chunk_size=n).start()
stream = Stream(bufsize=0.5, name='MNE-LSL-Player', stype='eeg', source_id='MNE-LSL').connect()

while True:
    previous = norm(stream.get_data(picks=ch)[0])
    time.sleep(0.5)
    now = norm(stream.get_data(picks=ch)[0])

    mne_previous = mne.io.RawArray(previous, info=data.copy().pick(ch).info, verbose=False)
    mne_now = mne.io.RawArray(now, info=data.copy().pick(ch).info, verbose=False)

    n_psd = mne_now.compute_psd(fmax=60, fmin=5)
    pred = model.predict(previous.reshape(1,len(ch), n))

    print(f"\n\nPredicted frequency band: {freqs[np.argmax(pred[-1])]}\
              \nReal Frequency band     : {freqs[np.argmax(n_psd.get_data()[-1])]}\
              \nCertainty               : {np.max(pred[-1])}")

    time.sleep(0.5)
