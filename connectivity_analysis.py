import matplotlib as mpl
import os
from mne_connectivity import spectral_connectivity_epochs
import mne
import matplotlib.pyplot as plt

### run preprocssing.py before this file
p="data/"


# Function to compute spectral connectivity using all different methods at once (PLI, dPLI, wPLI)
def spconn(epc, fmin, fmax, sfreq):
  return (
  spectral_connectivity_epochs(epc, method='pli', mode='multitaper', sfreq=sfreq, fmin= fmin, fmax=fmax, faverage=True, verbose=0),
  spectral_connectivity_epochs(epc, method='dpli', mode='multitaper', sfreq=sfreq, fmin=fmin, fmax=fmax, faverage=True, verbose=0),
  spectral_connectivity_epochs(epc, method='wpli', mode='multitaper', sfreq=sfreq, fmin= fmin, fmax=fmax, faverage=True, verbose=0),
  )


def run_conn(args):
    str_pat = f"{args['name']}_task-{args['task']}_dur-{args['start']}-{args['dur']+args['start']}_"
    data = mne.io.read_raw(p+args['file'], preload=True)
    evs, ids = mne.events_from_annotations(data)
    epc=mne.Epochs(data, event_repeated='merge')
    os.makedirs(p+args['newPath']+'/imgs/', exist_ok=True)
    img_p = p+args['newPath']+'/imgs/'

    ### alpha peak detection
    print("\n\n\n --- alpha peak\n\n\n")
    # Plot of the alpha band from the discrete data
    epc.compute_psd(fmin=8, fmax=12.9, method='welch',  n_jobs=8).plot(show=0).savefig(img_p+str_pat+'epoch-psd-alphaFreq.jpg')
    data.compute_psd(fmin=8, fmax=12.9,  n_jobs=8).plot(show=0).savefig(img_p+str_pat+'continuous-psd-alphaFreq.jpg')

    ### Brain activity analysis
    print("\n\n\n --- source reconstruction\n\n\n")
    dcov = mne.compute_covariance(epc, n_jobs=8)


    ### Connectivity analysis
    print("\n\n\n --- connectiviy analysis\n\n\n")
    # here, a figure is generatd with 3 plots, one for each method of computing spectral connectivity without a specific frequency band
    sf = int(data.info['sfreq'])
    pli, dpli, wpli = spconn(epc, args['fmin'], args['fmax'], sf)
    fig, axs = plt.subplots(1, 3, figsize=(10, 5), sharey=True)
    axs[0].imshow(pli.get_data("dense"), vmin=0, vmax=1)
    axs[0].set_title("PLI")
    axs[0].set_ylabel("Sensor 1")
    axs[0].set_xlabel("Sensor 2")
    axs[1].imshow(wpli.get_data("dense"), vmin=0, vmax=1)
    axs[1].set_title("wPLI")
    axs[1].set_xlabel("Sensor 2")
    im = axs[2].imshow(dpli.get_data("dense"), vmin=0, vmax=1)
    axs[2].set_title("dPLI")
    axs[2].set_xlabel("Sensor 2")
    plt.suptitle(f"Functional connectivity of {args['name']} among all frequency bands")
    fig.colorbar(im, ax=axs.ravel())
    fig.savefig(img_p+str_pat+"pli-dpli-wpli.jpg")

    # here, a figure is generated with a plot for each method of spectral connectivity and focusing on each of the 4 main frequency bands
    a_conn = spconn(epc, 8, 13 , sf)
    b_conn = spconn(epc, 12, 33, sf)
    t_conn = spconn(epc, 3.5, 8, sf)
    g_conn = spconn(epc, 25, 60, sf)
    fig1, axs2 = plt.subplots(3, 4, figsize=(10,10), sharey=True)

    for k, i, j in zip([0,1,2,3], [a_conn, b_conn, t_conn, g_conn], ["alpha", "beta", "theta", "gamma"]):
      axs2[0][k].imshow(i[0].get_data("dense"), vmin=0, vmax=1)
      axs2[0][k].set_title(j+" PLI")
      axs2[0][k].set_ylabel("Sensor 1")
      axs2[1][k].imshow(i[1].get_data("dense"), vmin=0, vmax=1)
      axs2[1][k].set_title(j+" dPLI")

      im = axs2[2][k].imshow(i[2].get_data("dense"), vmin=0, vmax=1)
      axs2[2][k].set_title(j+" wPLI")

      axs2[2][k].set_xlabel("Sensor 2")

    plt.suptitle(f"Functional connectivity of {args['name']} among specific frequency bands")
    fig1.colorbar(im, ax=axs2.ravel())
    fig1.savefig(img_p+str_pat+"connectiviy-allFreq.jpg")


    print("\n\n\n --- End of analysis ---\n",'.'*50,'\n\n')

def main():

    try:
        args = {
        "name":input("Enter the analyzed id (eg. the name of the participant): "),
        "file": input("Enter the filename to be analyzed: "),
        'task':input("Enter the EEG task: "),
        'start': float(input("Start time: ")),
        'dur': float(input("duration (in seconds): ")),
        "fmin": float(input("Minimum frequency: ")),
        "fmax": float(input("maximum frequency: ")),
        "newPath": input("Enter a directory name to save generated files: ")
        }
    except:
        print("Invalid entry")
        quit()

    run_conn(args)

main()
