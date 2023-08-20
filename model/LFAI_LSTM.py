#! /usr/bin/python3
import torch
import torch.nn as nn

# Define the LSTM model.
class LFAI_LSTM(nn.Module):
    def __init__(self, vocab_size:int, block_size:int, hidden_size:int, num_layers:int, device=torch.device('cpu')):
        super(LFAI_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.device = device
        self.embedding = nn.Embedding(block_size, hidden_size)
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