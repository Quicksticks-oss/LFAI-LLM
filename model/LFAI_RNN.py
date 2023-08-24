#! /usr/bin/python3
import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the RNN model.


class LFAI_RNN(nn.Module):
    def __init__(self, vocab_size: int, block_size: int, hidden_size: int, num_layers: int, device=torch.device('cpu'), dropout_p: int = 0.1, half: bool = False):
        super(LFAI_RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.dropout_p = dropout_p
        self.use_half = half

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size,
                          num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

        self.to(device)
        self.device = device

        # Precompute embeddings for all possible inputs
        self.embedding.weight.data = torch.randn(
            block_size + vocab_size, hidden_size).to(device)

    def forward(self, x, hidden=None):
        if hidden == None:
            hidden = self.init_hidden(1)
        embedded = self.embedding(x)
        # Adjust the dropout rate as needed
        embedded = F.dropout(embedded, p=self.dropout_p,
                             training=self.training)
        output, hidden = self.rnn(embedded, hidden)
        output = self.fc(output)
        return output, hidden

    def init_hidden(self, batch_size: int, inference: bool = False):
        weight = next(self.parameters()).data
        if inference == False:
            hidden = weight.new(self.num_layers, batch_size, self.hidden_size).zero_()
        else:
            noise = torch.randn(self.num_layers, batch_size, self.hidden_size)  # Generate random noise
            hidden = weight.new(self.num_layers, batch_size, self.hidden_size).zero_() + noise
        if self.use_half:
            return hidden.to(torch.float16)
        return hidden