import torch
from torch.utils.data import Dataset
from torch.nn import Module
import numpy as np

class LSTM_WAD(Module):
    def __init__(self, input_size, layers_size, output_size, num_lstm_layers, sequence_length, bidirectional = False, look_ahead = 0):
        super().__init__()

        self.input_size = input_size
        self.layer_size = layers_size
        self.output_size = output_size
        self.num_lstm_layers = num_lstm_layers
        self.sequence_length = sequence_length
        self.bidirectional = bidirectional
        self.look_ahead = look_ahead
    

        self.lstm_seq = (self.sequence_length + self.look_ahead) if self.bidirectional else self.sequence_length

        self.lstm_layers = None
        self.mid_layer = None
        self.out_layer = None

        self.create_network()

    def create_network(self):
        self.lstm_layers = torch.nn.LSTM(input_size = self.input_size, hidden_size = self.layer_size, num_layers = self.num_lstm_layers, batch_first = True, bidirectional = self.bidirectional)
        self.mid_layer = torch.nn.Sequential(
            torch.nn.Linear(in_features = (2*self.layer_size) if self.bidirectional else self.layer_size, out_features = self.layer_size),
            torch.nn.Tanh())
        self.dropout = torch.nn.Dropout(0.5)
        self.out_layer = torch.nn.Linear(in_features = self.layer_size, out_features = self.output_size)

    def forward(self,x):
        y_lstm,_ = self.lstm_layers(x)
        y = self.mid_layer(y_lstm[:,self.sequence_length - 1,:])
        y = self.dropout(y)
        y = self.out_layer(y)
        y = torch.nn.Sigmoid()(y)
        return y

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