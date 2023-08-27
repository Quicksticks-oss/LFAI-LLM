#! /usr/bin/python3
import torch.nn as nn
import torch

# Define the SI model.


class SIModel(nn.Module):
    def __init__(self, input_size: int, output_size: int = 16, hidden_size: int = 64) -> None:
        super(SIModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.input_layer = nn.Linear(self.input_size, self.hidden_size)
        self.hidden_layer = nn.Linear(self.hidden_size, self.hidden_size)
        self.output_layer = nn.Linear(self.hidden_size, self.output_size)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.input_layer(x)
        x = self.relu(self.hidden_layer(x))
        x = self.output_layer(x)
        return x


