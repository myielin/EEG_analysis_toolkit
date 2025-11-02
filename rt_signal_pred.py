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
model = load_model(p+"models/"+mn+".keras")

rt = Player(data, chunk_size=n).start()
stream = Stream(bufsize=0.5, name='MNE-LSL-Player', stype='eeg', source_id='MNE-LSL').connect()

fig, ax = plt.subplots()
ax.set_xlim(0, n)       # X range
ax.set_ylim(-5, 5)
ps, = ax.plot([], [], label="pred", color="tab:blue")
rs, = ax.plot([], [], label="real", color="tab:orange")
ax.legend()
now = norm(stream.get_data(picks=ch)[0][0])
pred = model.predict(now.reshape(1,1, n))[0][0]
xdata, ydata1, ydata2 = np.arange(n), np.array(pred),np.array(now)

def init():
    """Initialize the background of the animation"""
    ps.set_data([],[])
    rs.set_data([],[])
    return ps, rs

def anim(f):
    # get the new data
    now = norm(stream.get_data(picks=ch)[0][0])
    pred = norm(model.predict(now.reshape(1,1, n))[0][0])
    print("Predicted and real value: ",pred[-1], now[-2])

    # update the line plot:
    ps.set_data(xdata, pred)
    rs.set_data(xdata, now)

    ax.set_xlim(xdata[0], xdata[-1])  # scroll x-axis
    return ps, rs

ani = animation.FuncAnimation(fig, anim, interval=50, init_func=init, blit=True)
plt.show()

### uncomment the next line to save a gif
# ani.save(filename=p+"example.gif", writer="pillow")
