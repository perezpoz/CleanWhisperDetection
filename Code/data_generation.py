import os, glob
import pandas as pd
import torch
import torchaudio as ta
import torchaudio.functional as F
from tqdm import tqdm

def generate_dataset(speech_file_list, noise_files, dataset_output_path, fs_target, snr_values, utt_per_snr,min_silence_interval = 2, max_silence_interval = 4, min_noise_interval = 3, max_noise_interval = 8, train_test_label = None):

    num_speech_files = len(speech_file_list)

    if train_test_label == None:
        train_test_label = ''
    else:
        train_test_label = '_' + train_test_label

    # Generate noisy speech samples based on noise types.
    for noise_file in noise_files:

        noise_sig, sr = ta.load(noise_file)
        noise_basename = os.path.basename(noise_file)[:-4]
        # Keep only one of the channels
        if noise_sig.size(0) > 1:
            noise_sig = noise_sig[0,:]
        noise_sig = torch.reshape(noise_sig,(-1,))

        noise_sig = F.resample(noise_sig, sr, fs_target, resampling_method="kaiser_window")

        random_indices = torch.randperm(num_speech_files)

        file_indices_start = 0
        file_indices_end = utt_per_snr
        for _,snr in tqdm(enumerate(snr_values), total = len(snr_values)):

            noisy_signal = None
            noisy_signal = append_silence_interval(noisy_signal, fs = fs_target, min_silence_len = min_silence_interval, max_silence_len =  max_silence_interval)

            total_speech_samples = 0
            speech_energy = 0

            speech_start = []
            speech_end = []
            # Read speech files and concatenate with intermitent silences.
            for file_idx in tqdm(random_indices[file_indices_start:file_indices_end], total = utt_per_snr, leave=False):
                # Noisy speech recordings containing different speech utterances with silences in between and segments of noise affecting parts of the speech.
                audio_file = speech_file_list[file_idx]
                audio_sample, fs = ta.load(audio_file)
                audio_sample = torch.reshape(audio_sample, shape = (-1,))
                audio_sample = F.resample(audio_sample, fs, fs_target, resampling_method="kaiser_window")
                total_speech_samples += audio_sample.size(0)

                speech_energy += torch.sum(torch.square(audio_sample))

                speech_start.append(noisy_signal.size(0) / fs_target)

                noisy_signal = torch.cat((noisy_signal,audio_sample), dim = 0)
                
                speech_end.append(noisy_signal.size(0) / fs_target)

                noisy_signal = append_silence_interval(noisy_signal, fs = fs_target, min_silence_len = int(audio_sample.size(0) / fs_target), max_silence_len =  int(audio_sample.size(0) / fs_target) + 1)
                

            speech_energy /= total_speech_samples

            noise_sample_end = 0

            noise_start = []
            noise_end = []

            while(noise_sample_end < noisy_signal.size(0)):
                noise_sample_start = noise_sample_end

                noise_start.append(noise_sample_end / fs_target)

                # Add noise samples
                noise_interval = pick_noise_sample(noise_signal=noise_sig, fs = fs_target, min_noise_interval = min_noise_interval, max_noise_interval = max_noise_interval)
                
                noise_energy = torch.sum(torch.square(noise_interval)) / noise_interval.size(0)

                gain = (20 * torch.log10(speech_energy / noise_energy)) - snr
                linear_gain = torch.float_power(10, gain / 20)

                noise_interval = torch.multiply(linear_gain, noise_interval)

                noise_end.append((noise_sample_start + noise_interval.size(0)) / fs_target)

                noise_with_silence = append_silence_interval(noise_interval, fs = fs_target, min_silence_len = min_silence_interval, max_silence_len = max_silence_interval)

                noise_sample_end += noise_with_silence.size(0)

                diff_len = noisy_signal.size(0) - noise_sample_end

                if diff_len < 0:
                    noisy_signal[noise_sample_start:] += noise_with_silence[:diff_len]
                else:
                    noisy_signal[noise_sample_start:noise_sample_end] += noise_with_silence

            output_path = 'Noisy_' + noise_basename + '_' + str(snr) + '_dB' + train_test_label +'.wav'
            label_path = 'Noisy_' + noise_basename + '_' + str(snr) + '_dB' + train_test_label +'.csv'
            label_path_noise = 'Noisy_' + noise_basename + '_' + str(snr) + '_dB_Noise' + train_test_label +'.csv'
            output_path = os.path.join(dataset_output_path,output_path)
            label_path = os.path.join(dataset_output_path,label_path)
            label_path_noise = os.path.join(dataset_output_path,label_path_noise)
            ta.save(output_path, torch.reshape(noisy_signal,(1,-1)), sample_rate = fs_target)

            dict_speech = {'start':speech_start,'end':speech_end}
            dict_noise = {'start':noise_start, 'end':noise_end}
            df_speech = pd.DataFrame.from_dict(dict_speech)
            df_speech.to_csv(label_path)
            df_noise = pd.DataFrame.from_dict(dict_noise)
            df_noise.to_csv(label_path_noise)

            file_indices_start += utt_per_snr
            file_indices_end += utt_per_snr

def append_silence_interval(signal = None, fs = 16000, min_silence_len = 1, max_silence_len = 3):
    """
    Append a silence interval to the input signal.
    Params:
        - singal: Input signal. If None, return only the silence interval.
        - fs: Sampling rate.
        - min_silence_len: minimum length of the silence interval.
        - max_silence_len: maximum length of the silence interval.
    Return:
        - Input signal with silence concatenated at the end.
    """
    silence_length = torch.randint(low=min_silence_len, high=max_silence_len, size = (1,1))

    silence_len_samples = int(silence_length * fs)

    silence_interval = torch.zeros((silence_len_samples,))

    if signal == None:
        out_signal = silence_interval
    else:
        out_signal = torch.cat((signal,silence_interval), dim = 0)

    return out_signal

def pick_noise_sample(noise_signal, fs, min_noise_interval = 5, max_noise_interval = 10):
    """
    Pick a random interval from the input noise signal
    """

    noise_len = noise_signal.size(0)

    noise_interval_start = torch.randint(low = 0, high = noise_len - int(fs * max_noise_interval), size = (1,1))
    noise_interval_len = int(fs * torch.randint(low = min_noise_interval, high = max_noise_interval, size = (1,1)))

    noise_interval = noise_signal[noise_interval_start:noise_interval_start + noise_interval_len]
    return noise_interval

def separate_train_test_files(directory_list):
    """
    Separate several speakers as test files.
    """
    test_speakers = ['frf04', 'irf10', 'irf11', 'irf12', 'frm04','irm13', 'irm14', 'irm15', 'irm16']

    train_list = directory_list.copy()
    test_list = []
    for _,speaker_test in enumerate(test_speakers):

        for j,speaker_path in enumerate(train_list):

            if speaker_test in speaker_path:
                test_list.append(speaker_path)
                train_list.pop(j)
                break

    train_file_list = []
    test_file_list = []

    for _,speaker in enumerate(test_list):
        speaker_list = glob.glob(os.path.join(speaker,'*.wav'))
        test_file_list.extend(speaker_list)

    for _,speaker in enumerate(train_list):
        speaker_list = glob.glob(os.path.join(speaker,'*.wav'))
        train_file_list.extend(speaker_list)

    return train_file_list, test_file_list