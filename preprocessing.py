import mne
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from pandas import read_csv
import mne_icalabel


def set_stnd_mon(data, mon='standard_1020'):
  mne.channels.make_standard_montage(mon)
  data.set_montage(mon)
  return data

def create_csv_annotations(fname, dur=2, desc="mark"):
  anns_csv = read_csv(fname).values.reshape(-1)
  ann_arr = np.array([int(i[3:5]) + int(i[:2])*60 for i in anns_csv])

  return mne.Annotations(ann_arr, dur, desc)


def preprocess(data, lfreq=1, hfreq=80, notch=60, ica=True):
  data = data.filter(l_freq=lfreq, h_freq=hfreq, picks='all')
  data = data.notch_filter(notch, picks='all')
  if ica:
    c = mne.preprocessing.ICA(n_components=len(data.ch_names), max_iter="auto", method="infomax", fit_params=dict(extended=True))
    c.fit(data, picks='all')

    components = mne_icalabel.label_components(data, c, method='iclabel')
    print(f"\n\nICA components: {components['labels']}\nCertainty: {components['y_pred_proba']}")
    # component removal will be carried on if the component is classified as an artifact with more than 90% probability
    artifact_indexes = [i for i in range(len(data.ch_names)) if components['y_pred_proba'][i] > 0.9 and components['labels'][i] != 'brain']
    c.exclude = artifact_indexes
    print("\nRemoved artifacts:: ", artifact_indexes)

  return data

def test(ds):
    num = 1 if ds == "test" else 2
    p="data/"
    info = mne.create_info(ch_names = ['P4', 'P3', 'Fp2', 'Fp1'], sfreq = 200, ch_types='eeg')
    raw_train = read_csv(p+f"sampleEEG{num}.txt")[[' EXG Channel 0', ' EXG Channel 1', ' EXG Channel 2', ' EXG Channel 3']].values.T
    mne_train = mne.io.RawArray(raw_train, info)
    annotated = mne_train.set_annotations(create_csv_annotations(p+f"sampleEVS{num}.csv", 1))

    processed = set_stnd_mon(annotated, mon='standard_1020')
    processed = preprocess(annotated)

    processed.save(fname=p+f"data{num}.fif", overwrite=True)


def main():
    print(" -"*40)
    print(" "*10+"EEG PREPROCESSING")
    print(' -'*40)


    data = input("\nEnter the filename to be analyzed or leave blank to run the test data: ")
    if data == "":
        test("test")  # change the argument to "train" to preprocess the sample2 file
        quit()

    try:
        mne_raw = mne.io.read_raw("data/"+data)
        print(f"Loaded {data} data. Basic information:")
        print(mne_raw.info)

        ann = input("\nEnter the annotations filename (csv only). Leave blank if there are no annotations: ")
        if ann != "": mne_raw = mne_raw.set_annotations(create_csv_annotations("data/"+ann))

        mont = input("\nUse a standard montage [y/n]? (read the README file for help) ").lower()
        if "y" in str(mont):
            print("available montage sets: ", mne.channels.get_builtin_montages())
            monname = input("\nSpecify the montage name or leave blank to use the 10-20 system.\n")
            if monname == "": monname = "standard_1020"
            mne_raw = set_stnd_mon(mne_raw, mon=str(monname))

        res = input("\nResample the data? [y/n] ").lower()
        if "y" in res:
            v = int(input("\nHow many Hz? "))
            mne_raw = mne_raw.resample(v)

        print("\nSpecify preprocessing parameters or leave blank to use default values")
        lf = input("High-pass filter: ")
        if lf == "": lf = 0.1
        hf = input("Low-pass filter: ")
        if hf == "": hf = 80
        nt = input("Notch filter: ")
        if nt == "": nt = 60
        ica = input("Use ICA? [y/n] ").lower()
        bool = False if "n" in ica else True

        mne_raw = preprocess(mne_raw, lfreq=lf, hfreq=hf, notch=nt, ica=bool)

        print("\n- - - complete - - -\n")
        sv = input("Enter the filename to save the preprocessed data: ")
        mne_raw.save(fname="data/"+sv+".fif", overwrite=True)



    except:
        print("Invalid input. Make sure that you typed your entry correctly or the file name + extention and path are correct. See more in the README file")


main()
