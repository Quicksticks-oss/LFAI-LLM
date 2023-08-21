#! /usr/bin/python3
import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the LSTM model.
class LFAI_LSTM(nn.Module):
    def __init__(self, vocab_size:int, block_size:int, hidden_size:int, num_layers:int, device=torch.device('cpu')):
        super(LFAI_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.device = device
        self.embedding = nn.Embedding(block_size + vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.to(device)

    def forward(self, x, hidden):
        embedded = self.embedding(x)
        output, hidden = self.lstm(embedded, hidden)
        output = self.fc(output)
        return output, hidden

    def init_hidden(self, batch_size:int):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.num_layers, batch_size, self.hidden_size).zero_().to(self.device),
                  weight.new(self.num_layers, batch_size, self.hidden_size).zero_().to(self.device))
        return hidden
    
# Define the LSTM V2 model.
class LFAI_LSTM_V2(nn.Module):
    def __init__(self, vocab_size: int, block_size: int, hidden_size: int, num_layers: int, device: torch.device, dropout_p:int=0.1):
        super(LFAI_LSTM_V2, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.dropout_p = dropout_p

        self.embedding = nn.Embedding(block_size + vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

        self.to(device)
        self.device = device

        # Precompute embeddings for all possible inputs
        self.embedding.weight.data = torch.randn(block_size + vocab_size, hidden_size).to(device)

    def forward(self, x, hidden=None):
        if hidden == None:
            hidden = self.init_hidden(1)
        embedded = self.embedding(x)
        embedded = F.dropout(embedded, p=self.dropout_p, training=self.training)  # Adjust the dropout rate as needed
        output, hidden = self.lstm(embedded, hidden)
        output = self.fc(output)
        return output, hidden

    def init_hidden(self, batch_size: int, inference: bool=False):
        weight = next(self.parameters()).data
        if not inference:
            hidden = (weight.new(self.num_layers, batch_size, self.hidden_size).zero_(),
                    weight.new(self.num_layers, batch_size, self.hidden_size).zero_())
        else:
            noise = torch.randn(self.num_layers, batch_size, self.hidden_size)  # Generate random noise
            hidden = (weight.new(self.num_layers, batch_size, self.hidden_size).zero_() + noise,
                    weight.new(self.num_layers, batch_size, self.hidden_size).zero_() + noise)
        return hidden