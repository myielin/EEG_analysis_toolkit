# EEG Analysis Toolkit

A lightweight, modular toolkit for preprocessing, analyzing, visualizing, and exporting EEG data. Designed for researchers and engineers working with electrophysiological recordings, this repository provides reusable pipelines and utilities for common EEG workflows (filtering, artifact correction, epoching, feature extraction, and plotting) while remaining framework-agnostic and easy to integrate into existing projects.

This README gives an overview of the project, installation instructions, usage examples, conventions, and contribution guidelines.

---

Table of contents
- Project goals
- Features
- Repository layout
- Installation
- Quick start
- Typical workflows
  - Preprocessing
  - Epoching & averaging
  - Feature extraction
  - Visualization
- Command-line tools (if included)
- Configuration & file formats
- Tests
- Contributing
- License
- Contact

---


Features
- Signal processing: bandpass/notch filtering, resampling, removal of noisy components
- Artifact handling: ICA helpers, bad-channel detection and interpolation, blink/line-noise suppression utilities
- Epoching: stimulus-locked and response-locked epoch extraction with baseline correction
- Feature extraction: power spectral density (PSD), band-power, time–frequency summaries, ERP/ERSP support
- Visualization: channel plots, topographies (scalp maps), time-series overlays, PSD and spectrograms
- IO: read/write helpers for common EEG formats (EDF/BDF/BrainVision/MNE-compatible FIF/CSV) and simple export to NumPy/CSV
- Small utilities: montage handling, channel name normalization, reproducible pipeline wrappers

Repository layout
- /data/      - example scripts showing common workflows and datasets
- /eeg_toolkit/   - core Python package with preprocessing, io, analysis, visualization modules
  - preprocessing.py
  - io.py
  - epoching.py
  - features.py
  - viz.py
  - utils.py
- /tests/         - unit and integration tests
- requirements.txt
- environment.yml - (optional) conda environment file for reproducible setup
- README.md       - this file

Installation

Prerequisites
- Python 3.8+ recommended
- Common scientific packages: numpy, scipy, matplotlib, pandas
- Optional but recommended: mne, scikit-learn, seaborn, pyxdf (for XDF), joblib

Install with pip (recommended)
1. Clone the repository:
   git clone https://github.com/myielin/EEG_analysis_toolkit.git
   cd EEG_analysis_toolkit

2. Create a virtual environment (optional but recommended):
   python -m venv .venv
   source .venv/bin/activate   # macOS / Linux
   .\.venv\Scripts\activate    # Windows

3. Install dependencies:
   pip install -r requirements.txt

4. Install in editable/development mode:
   pip install -e .

Conda (if environment.yml is provided)
   conda env create -f environment.yml
   conda activate eeg-analysis

Quick start

Example: load, preprocess, epoch, and plot an ERP
```python
from eeg_toolkit.io import read_raw
from eeg_toolkit.preprocessing import bandpass, detect_bad_channels
from eeg_toolkit.epoching import make_epochs
from eeg_toolkit.viz import plot_erp

# 1) Read a raw file (format auto-detected)
raw = read_raw("data/sub-01_task-rest_eeg.edf")

# 2) Basic preprocessing: filter, downsample, detect bad channels
raw = bandpass(raw, low=1.0, high=40.0, inplace=False)
raw.resample(250)
bad_chs = detect_bad_channels(raw)
raw.info['bads'] = bad_chs
raw.interpolate_bads()

# 3) Epoch around events of interest
epochs = make_epochs(raw, event_id={"stim": 1}, tmin=-0.2, tmax=0.8, baseline=(None, 0))

# 4) Compute and plot ERP
erp = epochs.average()
plot_erp(erp, picks=["Cz", "Pz", "Fz"])
```

Typical workflows

Preprocessing
- Import raw EEG
- Inspect channels and metadata
- Apply notch filter for mains noise (50/60Hz)
- Bandpass to desired range (e.g., 1–40 Hz for ERPs)
- Downsample if necessary
- Detect and mark bad channels; interpolate
- Optional ICA for ocular/heartbeat artifact removal (helpers provided to fit/apply ICA and select components)

Epoching & averaging
- Define events and event IDs (stimulus, response)
- Extract epochs with specified tmin/tmax and baseline correction
- Reject epochs using amplitude thresholds or automated criteria
- Compute condition-wise average ERPs or single-trial summaries

Feature extraction
- Compute PSD via Welch or multitaper
- Extract band-power (delta/theta/alpha/beta/gamma)
- Time–frequency decomposition (STFT, wavelets) helpers for ERSP-like analyses
- Export feature matrices for machine learning models

Visualization
- Time-series plotting with event markers and channel selection
- Topographic maps (scalp plots) for single timepoints or time windows
- PSD and spectrogram plots
- Interactive plotting helpers (when supported by matplotlib backends)

Command-line tools (if provided)
- scripts/convert_to_numpy.py — convert raw recordings to numpy arrays plus event tables
- scripts/quick_erp.py — run a simple ERP pipeline on a specified dataset
(Check the /examples or /scripts directory for available CLIs)

Configuration & file formats
- Default configuration values (filter cutoffs, resample rates, epoch windows, rejection thresholds) live in eeg_toolkit/utils.py and can be overridden by passing parameters to functions.
- IO helpers try to detect common formats but prefer MNE-compatible structures when available. If you need support for a format not yet included, please open an issue or contribute a reader/writer.

Tests
- Tests live under /tests
- Run tests with pytest:
  pytest -q

Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository and create a feature branch.
2. Write tests for new functionality.
3. Make changes and keep commits focused and atomic.
4. Run tests and linters (if configured).
5. Open a pull request describing your changes and rationale.

Please follow the code style in the project (PEP8) and include docstrings for public functions. If you plan a large change, open an issue first to discuss design choices.

License
Specify a license (e.g., MIT, BSD, or other). If no license file exists, add one or update this section accordingly.


Acknowledgements
- Built with numpy, scipy, matplotlib, and optionally MNE
- Derived ideas and patterns from community EEG toolkits and best practices in cognitive electrophysiology

Contact
Repository: https://github.com/myielin/EEG_analysis_toolkit
Author / maintainer: myielin
