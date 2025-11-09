import numpy as np
import mne
from numpy.lib.stride_tricks import sliding_window_view
from os import makedirs



p = 'data/'
def norm(x):
    normX = (x-np.mean(x)) / np.std(x)
    return normX


def unnorm(x):
    normX = x * np.std(x) + np.mean(x)
    return normX

def split_seq(seq, ch, n, step, normalize=False):
    data = seq.get_data(picks=ch)

    if normalize: data = norm(data)

    sigs = sliding_window_view(data, window_shape=n, axis=1).transpose(1, 0, 2)
    sigs = sigs[::step]

    freqs = np.empty(sigs.shape[0])
    for i in range(sigs.shape[0]):
      mne_i = mne.io.RawArray(sigs[i], info=seq.copy().pick(ch).info, verbose = False)
      psd = mne_i.compute_psd(fmax=60, fmin=5, verbose = False)
      freqs[i] = psd.freqs[np.argmax(psd.get_data()[-1])]

    return sigs, freqs

def sv_model(model):
    s = input("\nsave the trained model? [y/n] ")
    if 'y' in s:
        sname = input("Enter the file name (without the .keras extention): ")
        makedirs(p+"models/", exist_ok=True)
        model.save(p+"models/"+sname+".keras")


def init_script(str1, str2):
    print(" -"*40)
    print(" "*10+str1)
    print(' -'*40)

    fname = input(f"\nEnter the EEG filename {str2}. Leave blank to use the sample data: ")

    try:
        data = mne.io.read_raw(p+fname) if fname != "" else mne.io.read_raw_fif(p+"data1.fif")
    except:
        print("\nFile not found. Make sure both path and file name are correct")
        quit()

    n = int(data.info['sfreq']//2)
    chs = input("\nEnter the set of channels to be included, separated by space. If left empty the channel Fp1 is selected. ")
    ch = ["Fp1"] if chs == "" else chs.split(" ")

    return p, fname, data, n, ch
