import os, sys, glob
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score

def main():
    dataset_path = os.path.normpath('./Dataset/')

    true_files_path = os.path.join(dataset_path, '*dB_Test.csv')
    true_files = glob.glob(true_files_path)

    true_label_10db = np.empty((1,))
    true_label_5db = np.empty((1,))
    true_label_0db = np.empty((1,))

    py_label_10db = np.empty((1,))
    py_label_5db = np.empty((1,))
    py_label_0db = np.empty((1,))

    for true_file in true_files:
        pyannote_file = true_file[:-4] + '_Pyannote.csv'

        true_timeslots = pd.read_csv(true_file)
        true_start = true_timeslots['start']
        true_end = true_timeslots['end']
        pyannote_timeslots = pd.read_csv(pyannote_file)
        py_start = pyannote_timeslots['start']
        py_end = pyannote_timeslots['end']

        timestep = 0.02
        
        true_labels = np.asarray(sample_timeslots(true_start, true_end, timestep)).reshape((-1,))
        py_labels = np.asarray(sample_timeslots(py_start, py_end, timestep)).reshape((-1,))

        if true_labels.shape[0] > py_labels.shape[0]:
            py_labels = np.pad(py_labels,[0,true_labels.shape[0] - py_labels.shape[0]])
        elif py_labels.shape[0] > true_labels.shape[0]:
            true_labels = np.pad(true_labels,[0,py_labels.shape[0] - true_labels.shape[0]])

        if '10_dB' in true_file:
            true_label_10db = np.concatenate((true_label_10db,true_labels), axis = 0)
            py_label_10db = np.concatenate((py_label_10db,py_labels), axis = 0)
        elif '5_dB' in true_file:
            true_label_5db = np.concatenate((true_label_5db,true_labels), axis = 0)
            py_label_5db = np.concatenate((py_label_5db,py_labels), axis = 0)
        elif '0_dB' in true_file:
            true_label_0db = np.concatenate((true_label_0db,true_labels), axis = 0)
            py_label_0db = np.concatenate((py_label_0db,py_labels), axis = 0)
        else:
            print('Incorrect SNR value')
            exit()

    true_label_10db = true_label_10db[1:]
    py_label_10db = py_label_10db[1:]
    true_label_5db = true_label_5db[1:]
    py_label_5db = py_label_5db[1:]
    true_label_0db = true_label_0db[1:]
    py_label_0db = py_label_0db[1:]


    f1_10db = f1_score(true_label_10db, py_label_10db)
    f1_5db = f1_score(true_label_5db, py_label_5db)
    f1_0db = f1_score(true_label_0db, py_label_0db)

    acc_10db = accuracy_score(true_label_10db, py_label_10db)
    acc_5db = accuracy_score(true_label_5db, py_label_5db)
    acc_0db = accuracy_score(true_label_0db, py_label_0db)

    print(f1_10db)
    print(acc_10db)
    print(f1_5db)
    print(acc_5db)
    print(f1_0db)
    print(acc_0db)


def sample_timeslots(time_start, time_end, timestep):

    """
    Convert time start/end labels extracted by pyannote into an array of ones or zeros depending on the VAD output.
    """

    labels = []

    current_time = 0

    start_idx = 0
    end_idx = 0

    num_slots = len(time_end)

    while(end_idx < num_slots):

        if (start_idx < num_slots):
            start_idx += 1 if (current_time > time_start[start_idx]) else 0

        if (current_time > time_end[end_idx]):
            end_idx += 1
        
        if start_idx == end_idx:
            labels.append(0)
        elif start_idx > end_idx:
            labels.append(1)
        else:
            print('Something went wrong')
            exit()
        
        current_time += timestep
    
    return labels

if __name__ == "__main__":
    main()