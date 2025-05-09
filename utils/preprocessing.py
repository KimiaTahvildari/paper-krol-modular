import wfdb
import numpy as np
import pywt
from config import FS, SEGMENT_LENGTH, AFIB_LABEL

def load_ecg(record_path, lead_index=0):
    """
    Load ECG signal from a record path.
    """
    record = wfdb.rdrecord(record_path)
    ecg_signal = record.p_signal[:, lead_index]
    return ecg_signal, record.fs

def load_rhythm_annotations(record_path):
    """
    Load rhythm annotations and map them to numeric labels.
    """
    ann = wfdb.rdann(record_path, 'atr')
    rhythm_map = {
        '(N': 0,
        '(AFIB': 1,
        '(AFL': 2,
        '(J': 3
    }
    return [(ann.sample[i], rhythm_map[aux.strip()]) 
            for i, aux in enumerate(ann.aux_note) if aux.strip() in rhythm_map]

def segment_by_rhythm(signal, rhythm_ann, fs=FS):
    """
    Segment the ECG signal into AFIB and non-AFIB segments.
    """
    afib_segments, non_afib_segments = [], []
    intervals = [(rhythm_ann[i][0], rhythm_ann[i+1][0], rhythm_ann[i][1]) for i in range(len(rhythm_ann) - 1)]

    for i in range(0, len(signal) - SEGMENT_LENGTH + 1, SEGMENT_LENGTH):
        segment = signal[i:i + SEGMENT_LENGTH]
        if np.std(segment) < 0.01:
            continue  # skip flat segments
        label = 0
        for start, end, r_label in intervals:
            if r_label == AFIB_LABEL and start <= i and i + SEGMENT_LENGTH <= end:
                label = 1
                break
        if label == 1:
            afib_segments.append(segment)
        else:
            non_afib_segments.append(segment)

    return afib_segments, non_afib_segments

def save_scalogram_as_array(signal, save_path):
    """
    Convert the ECG signal to a scalogram using CWT and save as a .npy file.
    """
    widths = np.arange(1, 128)
    cwtmatr, _ = pywt.cwt(signal, widths, 'mexh')
    scalogram = np.abs(cwtmatr).astype(np.float32)
    np.save(save_path, scalogram)
