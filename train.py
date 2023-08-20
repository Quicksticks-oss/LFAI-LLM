#! /usr/bin/python3
from model.LFAI_LSTM import LFAI_LSTM
from pathlib import Path
from tqdm import tqdm
from utils import *
import torch.nn as nn
import argparse
import datetime
import random
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
        self.chars = None
        self.model = None
        self.text = None
        self.data = None

    def count_parameters(self):
        self.params = round(sum(p.numel()
                            for p in self.model.parameters()) / 1_000_000, 2)
        return self.params

    def get_batch(self, split):
        # generate a small batch of data of inputs x and targets y
        size = random.randint(1,self.context_length)
        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(len(data) - size, (self.batch_size,))
        x = torch.stack([data[i:i+size] for i in ix])
        y = torch.stack([data[i+1:i+size+1] for i in ix])
        x, y = x.to(self.device), y.to(self.device)
        return x, y

    def load_dataset(self):
        if self.dataset.exists():
            if self.dataset.is_dir():
                files = os.listdir(self.dataset)
                self.text = ''
                if len(files) > 0:
                    for file in files:
                        with open(self.dataset.joinpath(file), 'r') as f:
                            self.text += repr(f.read())
                else:
                    raise IOError('Dataset does not contain any files!')
            elif self.dataset.is_file():
                with open(self.dataset) as f:
                    self.text = repr(f.read())
        else:
            raise IOError('Dataset does not exist!')

    def load_tokenizer(self):
        if len(self.text) > 0:
            self.chars = sorted(list(set(self.text)))
            self.vocab_size = len(self.chars)
            # Create a mapping from characters to integers
            stoi = {ch: i for i, ch in enumerate(self.chars)}
            itos = {i: ch for i, ch in enumerate(self.chars)}
            # encoder: take a string, output a list of integers
            self.encode = lambda s: [stoi[c] for c in s]
            # decoder: take a list of integers, output a string
            self.decode = lambda l: ''.join([itos[i] for i in l])
        else:
            raise Exception(
                'Could now load tokenizer due to lack of data in dataset.')

    def convert_dataset(self):
        self.data = torch.tensor(self.encode(self.text), dtype=torch.long)
        n = int(1*len(self.data))  # first 90% will be train, rest val
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

            # Initialize the hidden state.
            hidden = self.model.init_hidden(self.batch_size)
            size = ((self.train_data.size(0)-1) - self.context_length) - (self.context_length*self.batch_size)
            td = tqdm(range(0, size), postfix='training... ] ',
                      dynamic_ncols=True)

            for _ in td:
                # Gets batch size.
                inputs_batch, targets_batch = self.get_batch('train')

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

                if _ % 8 == 0:
                    description = f'[ epoch: {epoch}, loss: {loss.item():.4f}'
                    tqdm.set_description(td, desc=description)
                    if _ % 128 == 0:
                        self.save('current')
            print()
            self.save()

    def save(self, name: str = None):
        create_folder_if_not_exists('weights')
        if name == None:
            name = self.save_file
        save_out = {
            'vocab_size': self.vocab_size,
            'context_length': self.context_length,
            'hidden_size': self.hiddensize,
            'num_layers': self.numlayers,
            'chars': self.chars,
            'state_dict': self.model.state_dict()
        }
        torch.save(save_out, Path('weights').joinpath(name+'.pth'))


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
    epochs = int(args.epochs)
    numlayers = int(args.numlayers)
    hiddensize = int(args.hiddensize)
    contextsize = int(args.contextsize)
    batch_size = int(args.batchsize)
    learningrate = float(args.learningrate)
    half = bool(args.half)

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
