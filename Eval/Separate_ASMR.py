import os, sys, glob
import torch
import torchaudio as ta
from torch.utils.data import DataLoader


from Baseline.DNN_WAD import LSTM_WAD, Whisper_sequence
from Features.rasta import rastaplp

from librosa.feature import delta

import pickle
import matplotlib.pyplot as plt

if os.name == 'nt':
    ta.set_audio_backend('soundfile')
else:
    ta.set_audio_backend('sox_io')

def main():

    # Initialise RASTA parameters
    num_rasta = 19
    num_features = 3 * num_rasta

    #Initialise LSTM parameters and load models
    sequence_length = 30

    asmr_path = os.path.normpath('./Youtube-DL/')
    model_path = os.path.normpath('./Checkpoints/LSTM_64_Clean_LongDS.pt')
    scaler_path = os.path.normpath('./Checkpoints/LSTM_Scaler.sav')

    model = LSTM_WAD(input_size = num_features, layers_size = 64, output_size = 1, num_lstm_layers = 2, sequence_length = sequence_length)
    model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
    model.eval()

    scaler = pickle.load(open(scaler_path, 'rb'))

    # Decision threshold
    opt_th = 0.5167811512947083

    audio_files = glob.glob(os.path.join(asmr_path, '**','*.wav'), recursive = True)

    

    for file in audio_files:
        
        if ('Noise' in file) or ('Speech' in file):
            continue

        noise_file = file[:-4] + '_NoiseTune.wav'
        speech_file = file[:-4] + '_SpeechTune.wav'

        audio_signal,sr = ta.load(file)

        audio_signal = ta.functional.resample(audio_signal, sr, 16000)
        sr = 16000
        hop_size = int(0.02 * sr)

        audio_mono = audio_signal[0,:]

        # Extract RASTA features
        rasta_sig = rastaplp(x = audio_mono.numpy(), fs = sr, win_time = 0.04, hop_time = 0.02, dorasta = True, modelorder = num_rasta - 1)

        rasta_delta = delta(rasta_sig)

        rasta_delta_delta = delta(rasta_delta)

        features = torch.cat((torch.from_numpy(rasta_sig), torch.from_numpy(rasta_delta), torch.from_numpy(rasta_delta_delta)), dim = 0)

        features = torch.transpose(features,0,1)

        num_samples = features.shape[0]

        features = scaler.transform(features)

        # Convert feature matrix into dataset - loader
        labels = torch.zeros(num_samples,1)
        feat_ds = Whisper_sequence(data = features, labels = labels, sequence_length = sequence_length, segment_lengths = [num_samples - sequence_length + 1])
        feat_dl = DataLoader(dataset = feat_ds, shuffle = False)

        idx = 0
        for x,_ in feat_dl:
            y = model(x.type(torch.FloatTensor))
            y_pred = 1 if y > opt_th else 0#[1 if y_test > opt_th else 0 for y_test in y]
            
            labels[sequence_length - 1 + idx] = y_pred
            idx += 1

        # Assign labels to signal length
        sig_flag = torch.zeros(audio_mono.size(0),1)
        start_idx = hop_size
        end_idx = start_idx + hop_size
        for i in range(len(labels)):
            lab = labels[i]
            sig_flag[start_idx:end_idx] = int(lab)
            start_idx += hop_size
            end_idx += hop_size

        sig_flag[end_idx:] = int(sig_flag[end_idx-1])
        sig_flag = torch.reshape(sig_flag,(-1,))

        # Separate noise segments from speech segments
        noise_mask = (sig_flag == 0)
        speech_mask = (sig_flag == 1)
        noise_signal = audio_mono[noise_mask]
        speech_signal = audio_mono[speech_mask]

        ta.save(noise_file, torch.reshape(noise_signal,(1,-1)), sr)
        ta.save(speech_file, torch.reshape(speech_signal,(1,-1)), sr)

        print('File ' + os.path.basename(file) + ' has been split.')

    
if __name__ == "__main__":
    main()