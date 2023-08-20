#! /usr/bin/python3
from model.LFAI_LSTM import LFAI_LSTM
from pathlib import Path
from tqdm import tqdm
from utils import *
import torch.nn as nn
import argparse
import datetime
import torch
import os

# Main trainer class.


class Trainer:
    def __init__(self, name, dataset, epochs, context_length, numlayers, hiddensize, batch_size=12, learning_rate=0.001, half: bool = False) -> None:
        current_date = str(datetime.datetime.now().date()).replace('-', '')
        self.name = name
        self.dataset = Path(dataset)
        self.epochs = epochs
        self.batch_size = batch_size
        self.numlayers = numlayers
        self.hiddensize = hiddensize
        self.learning_rate = learning_rate
        self.context_length = context_length
        self.half = half
        self.params = 0
        self.save_file = f'{name}-{self.params}M-{current_date}-{numlayers}-{hiddensize}-ctx{context_length}'
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.vocab_size = 0
        self.model = None
        self.text = None
        self.data = None

    def count_parameters(self):
        self.params = round(sum(p.numel()
                            for p in self.model.parameters()) / 1_000_000, 2)
        return self.params

    def load_dataset(self):
        if self.dataset.exists():
            if self.dataset.is_dir():
                files = os.listdir(self.dataset)
                self.text = ''
                if len(files) > 0:
                    for file in files:
                        with open(self.dataset.joinpath(file), 'r') as f:
                            self.text += f.read()
                else:
                    raise IOError('Dataset does not contain any files!')
            elif self.dataset.is_file():
                with open(self.dataset) as f:
                    self.text = f.read()
        else:
            raise IOError('Dataset does not exist!')

    def load_tokenizer(self):
        if len(self.text) > 0:
            chars = sorted(list(set(self.text)))
            self.vocab_size = len(chars)
            # Create a mapping from characters to integers
            stoi = {ch: i for i, ch in enumerate(chars)}
            itos = {i: ch for i, ch in enumerate(chars)}
            # encoder: take a string, output a list of integers
            self.encode = lambda s: [stoi[c] for c in s]
            # decoder: take a list of integers, output a string
            self.decode = lambda l: ''.join([itos[i] for i in l])
        else:
            raise Exception(
                'Could now load tokenizer due to lack of data in dataset.')

    def convert_dataset(self):
        self.data = torch.tensor(self.encode(self.text), dtype=torch.long)
        n = int(0.9*len(self.data))  # first 90% will be train, rest val
        self.train_data = self.data[:n]
        self.val_data = self.data[n:]
        del (self.text)

    def create_model(self):
        self.model = LFAI_LSTM(self.vocab_size, self.context_length,
                               self.hiddensize, self.numlayers, self.device)
        if self.half:
            self.model.half()
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate)

    def train(self):
        # Training loop
        for epoch in range(self.epochs):
            self.model.train()  # Puts the model into training mode.

            # Initialize the hidden state
            hidden = self.model.init_hidden(self.batch_size)

            size = ((self.train_data.size(0)-1) - self.context_length) - \
                (self.context_length*self.batch_size)
            td = tqdm(range(0, size), postfix='training... ] ',
                      dynamic_ncols=True)

            for _ in td:
                # Prepare the inputs and targets for the current batch
                inputs_batch = self.train_data[_:_ + self.context_length * self.batch_size].view(
                    self.batch_size, self.context_length).to(self.device)
                targets_batch = self.train_data[_ + 1:_ + self.context_length * self.batch_size + 1].view(
                    self.batch_size, self.context_length).to(self.device)

                if self.half:  # Convert to half
                    inputs_batch = inputs_batch.to(torch.int16)
                    targets_batch = targets_batch.to(torch.int16)

                self.optimizer.zero_grad()

                outputs, hidden = self.model(inputs_batch, hidden)
                output_flat = outputs.view(-1, self.vocab_size)
                loss = self.criterion(
                    outputs.view(-1, self.vocab_size), targets_batch.view(-1))

                loss.backward()
                self.optimizer.step()

                # Detach hidden state to prevent backpropagation through time
                hidden = tuple(h.detach() for h in hidden)

                description = f'[ epoch: {epoch}, loss: {loss.item():.4f}'
                tqdm.set_description(td, desc=description)
                if _ % 16 == 0:
                    self.save('current.pt')
            print()
            self.save()

    def save(self, name: str = None):
        create_folder_if_not_exists('weights')
        if name == None:
            name = self.save_file
        torch.save(self.model.state_dict(), Path('weights').joinpath(name))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default='LFAI',
                        help="Specify a model name.", required=True)
    parser.add_argument("--epochs", default=1,
                        help="Specify how many epochs the model should train with.", required=False)
    parser.add_argument("--dataset", default='input.txt',
                        help="Specify the dataset the model will train on, this can be a file or folder.", required=True)
    parser.add_argument("--numlayers", default=6,
                        help="Specify how many layers the rnn layer will have.", required=False)
    parser.add_argument("--hiddensize", default=128,
                        help="Specify how large the hidden layer size will be.", required=False)
    parser.add_argument("--contextsize", default=512,
                        help="Specify the max length of the input context field.", required=False)
    parser.add_argument("--batchsize", default=32,
                        help="Specify how much data will be passed at one time.", required=False)
    parser.add_argument("--learningrate", default=0.001,
                        help="Specify how confident the model will be in itself.", required=False)
    parser.add_argument("--half", default=False,
                        help="Specify if the model should use fp16 (Only for GPU).", required=False)

    args = parser.parse_args()

    name = args.name
    dataset = args.dataset
    epochs = args.epochs
    numlayers = args.numlayers
    hiddensize = args.hiddensize
    contextsize = args.contextsize
    batch_size = args.batchsize
    learningrate = args.learningrate
    half = args.half

    trainer = Trainer(name, dataset, epochs,
                      contextsize, numlayers, hiddensize,
                      batch_size, learningrate, half)
    print('Loading dataset...')
    trainer.load_dataset()
    trainer.load_tokenizer()
    trainer.convert_dataset()
    print('Creating model...')
    trainer.create_model()
    print(f'Created model with {trainer.count_parameters()}M params...')
    print('Training model...')
    trainer.train()
    print('Training complete...')


if __name__ == '__main__':
    main()
