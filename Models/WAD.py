import torch

class MLP_WAD(torch.nn.Module):
    """
    Feedforward network for detecting whisper in noise.
    """
    def __init__(self, input_size, layer_sizes, output_size):
        super().__init__()

        self.input_size = input_size
        self.layer_sizes = layer_sizes
        self.output_sizes = output_size
        self.layers = None
        self.network = None

        self.create_network()


    def create_network(self):
        layers=[]

        layers.append(torch.nn.Linear(in_features = self.input_size, out_features = self.layer_sizes[0]))
        layers.append(torch.nn.ReLU())

        for layer in range(len(self.layer_sizes[1:])):
            layers.append(torch.nn.Linear(in_features = self.layer_sizes[layer], out_features = self.layer_sizes[layer+1]))
            layers.append(torch.nn.ReLU())
        
        layers.append(torch.nn.Linear(in_features = self.layer_sizes[-1], out_features = self.output_sizes))
        layers.append(torch.nn.Sigmoid())
    
        self.network = torch.nn.Sequential(*layers)
        self.layers = layers

    def forward(self, x):
        #y = self.network(x)
        y = x
        for l in self.layers:
            y = l(y)
        return y


class LSTM_WAD(torch.nn.Module):
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