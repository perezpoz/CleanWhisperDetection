import torch
from torch.utils.data import Dataset
import numpy as np

class Whisper_dataset(Dataset):
    def __init__(self, data, labels):
        super().__init__()
        self.data = data
        self.labels = labels
        


        assert(self.data.shape[0] == self.labels.shape[0])

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        return self.data[index,:], self.labels[index]

class Whisper_sequence(Dataset):
    def __init__(self, data, labels, sequence_length, segment_lengths):
        super().__init__()
        self.data = data
        self.labels = labels
        self.sequence_length = sequence_length

        self.segment_lengths = segment_lengths

        self.data_indices = None

        self.generate_sequence_indices()

    def __len__(self):
        return len(self.data_indices)

    def __getitem__(self, index):

        idx_data_start = self.data_indices[index] - self.sequence_length + 1
        idx_data_end = self.data_indices[index] + 1

        data_frame = self.data[idx_data_start:idx_data_end,:]
        label = self.labels[self.data_indices[index]]

        return data_frame, label
    
    def generate_sequence_indices(self):
        self.data_indices = []
        for _,len in enumerate(self.segment_lengths):
            segment_indices = np.arange(self.sequence_length, len)

            if not self.data_indices:
                self.data_indices.extend(segment_indices.tolist())
            else:
                segment_indices += (self.data_indices[-1] + 1)
                self.data_indices.extend(segment_indices.tolist())

class Bidir_whisper_sequence(Dataset):
    def __init__(self, data, labels, sequence_length, look_ahead, segment_lengths):
        super().__init__()
        self.data = data
        self.labels = labels
        self.sequence_length = sequence_length
        self.look_ahead = look_ahead

        self.segment_lengths = segment_lengths

        self.data_indices = None

        self.generate_bidir_sequence_indices()

    def __len__(self):
        return len(self.data_indices)

    def __getitem__(self, index):

        idx_data_start = self.data_indices[index] - self.sequence_length + 1
        idx_data_end = self.data_indices[index] + self.look_ahead + 1

        data_frame = self.data[idx_data_start:idx_data_end,:]
        label = self.labels[self.data_indices[index]]

        return data_frame, label
    
    def generate_bidir_sequence_indices(self):
        self.data_indices = []
        for _,len in enumerate(self.segment_lengths):
            segment_indices = np.arange(self.sequence_length, len - self.look_ahead)

            if not self.data_indices:
                self.data_indices.extend(segment_indices.tolist())
            else:
                segment_indices += (self.data_indices[-1] + 1)
                self.data_indices.extend(segment_indices.tolist())
            
class early_stopper():
    def __init__(self,max_iter_higher_loss):
        self.max_iter_higher_loss = max_iter_higher_loss
        self.count_higher_loss = 0
        self.current_min = np.Inf

    def update_checkpoint(self, new_loss):
        if new_loss < self.current_min:
            self.count_higher_loss = 0
            self.current_min = new_loss
            return True
        else:
            self.count_higher_loss += 1
            return False

    def stop_training(self):
        
        if self.count_higher_loss > self.max_iter_higher_loss:
            return True
        return False