#! /usr/bin/python3
import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the Linear model.


class LFAI_Linear(nn.Module):
    def __init__(self, vocab_size: int, block_size: int, hidden_size: int, device: torch.device = torch.device('cpu'), dropout_p: float = 0.1, half: bool = False):
        super(LFAI_Linear, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.dropout_p = dropout_p
        self.use_half = half

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, vocab_size)
        )

        self.to(device)
        self.device = device

        # Precompute embeddings for all possible inputs
        self.embedding.weight.data = torch.randn(
            block_size + vocab_size, hidden_size).to(device)

    def forward(self, x):
        embedded = self.embedding(x)
        # Adjust the dropout rate as needed
        embedded = F.dropout(embedded, p=self.dropout_p,
                             training=self.training)
        output = self.fc_layers(embedded)
        return output
