#! /usr/bin/python3
import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the Linear model.


class LFAI_LINEAR(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, device: torch.device = torch.device('cpu'), dropout_p: float = 0.1, half: bool = False):
        super(LFAI_LINEAR, self).__init__()
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p
        self.use_half = half

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)

        self.to(device)
        self.device = device

    def forward(self, x):
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = F.relu(self.fc3(x))
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        output = self.fc4(x)
        return output
