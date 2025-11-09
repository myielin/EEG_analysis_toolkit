from mne_realtime import MockRtClient,  RtEpochs
from mne_lsl.player import PlayerLSL as Player
from mne_lsl.stream import StreamLSL as Stream
from mne_lsl.stream import EpochsStream as eStream
import mne
import warnings
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from keras.saving import load_model
import time
import numpy as np
from functions import init_script, norm
warnings.filterwarnings("ignore")

p, fname, data, n, ch = init_script("REAL-TIME PREDICTION", "to simulate real-time data acquisition from")
mn =  input("Enter the trained model name without the .keras extension:  ")
if mn == "": mn = "sampleSeries"
model = load_model(p+"models/"+mn+".keras")

rt = Player(data, chunk_size=n).start()
stream = Stream(bufsize=0.5, name='MNE-LSL-Player', stype='eeg', source_id='MNE-LSL').connect()
l = len(ch)
fig, ax = plt.subplots()
ax.set_xlim(0, n)       # X range
ax.set_ylim(-5, 5)
ps, = ax.plot([], [], label="pred", color="tab:blue")
rs, = ax.plot([], [], label="real", color="tab:orange")
ax.legend()
now = norm(stream.get_data(picks=ch)[0])
pred = model.predict(now.reshape(1,l, n))[0]
xdata, ydata1, ydata2 = np.arange(n), np.array(pred[0]),np.array(now[0])

def init():
    """Initialize the background of the animation"""
    ps.set_data([],[])
    rs.set_data([],[])
    return ps, rs

def anim(f):
    # get the new data
    previous = norm(stream.get_data(picks=ch)[0])
    time.sleep(0.5)
    now = norm(stream.get_data(picks=ch)[0])
    pred = model.predict(previous.reshape(1,l, n))[0]
    print("Some predicted and real values: ",pred[-1][0:4], now[-1][0:4])

    # update the line plot:
    ps.set_data(xdata, pred[0])
    rs.set_data(xdata, now[0])

    ax.set_xlim(xdata[0], xdata[-1])  # scroll x-axis
    return ps, rs

ani = animation.FuncAnimation(fig, anim, interval=100, init_func=init, blit=True).save(filename=p+"example.gif", writer="pillow")
### uncomment the next line to save a gif
plt.show()
