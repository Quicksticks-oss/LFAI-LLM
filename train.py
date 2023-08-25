#! /usr/bin/python3
from model.LFAI_LSTM import LFAI_LSTM
from model.LFAI_LSTM import LFAI_LSTM_V2
from model.LFAI_Linear import LFAI_Linear
from model.LFAI_GRU import LFAI_GRU
from model.LFAI_RNN import LFAI_RNN
from tokenizers.tokenizer_v3 import Tokenizer_V3
from tokenizers.tokenizer_v2 import Tokenizer_V2
from tokenizers.tokenizer_v1 import Tokenizer_V1
from pathlib import Path
from tqdm import tqdm
from utils import *
import torch.nn as nn
import argparse
import datetime
import random
import torch
import time
import os
import io

# Main trainer class.


class Trainer:
    def __init__(self, name, dataset, epochs, context_length, numlayers, hiddensize,
                 batch_size=12, learning_rate=0.001, half: bool = False, version: int = 2,
                 load: str = "", network: str = 'lstm', graph: bool = False, savelast: bool = False) -> None:
        self.current_date = str(
            datetime.datetime.now().date()).replace('-', '')
        self.name = name
        self.dataset = Path(dataset)
        self.graph = graph
        self.epochs = epochs
        self.savelast = savelast
        self.version = version
        self.batch_size = batch_size
        self.numlayers = numlayers
        self.hiddensize = hiddensize
        self.learning_rate = learning_rate
        self.context_length = context_length
        self.network = network.lower()
        self.half = half
        self.params = 0
        self.save_file = f'{name}-{self.params}M-{self.current_date}-{numlayers}-{hiddensize}-ctx{context_length}'
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {self.device}')
        self.ds_files = []
        self.vocab_size = 0
        self.load = load
        self.chars = None
        self.model = None
        self.text = None
        self.data = None

    def encode_and_combine_chunks(self, chunk_size=10000000):
        chunks = [self.text[i:i+chunk_size]
                  for i in range(0, len(self.text), chunk_size)]
        encoded_chunks = [self.tokenizer.encode(chunk) for chunk in chunks]
        combined_data = torch.tensor(
            [val for chunk in encoded_chunks for val in chunk], dtype=torch.long)
        return combined_data

    def count_parameters(self):
        self.params = round(sum(p.numel()
                            for p in self.model.parameters()) / 1_000_000, 2)
        self.save_file = f'{self.name}-{self.params}M-{self.current_date}-{self.numlayers}-{self.hiddensize}-ctx{self.context_length}'
        return self.params

    def get_batch(self, split: str = 'train'):
        # Generate a small batch of data of inputs x and targets y
        size = random.randint(1, self.context_length)
        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(len(data) - size, (self.batch_size,))

        x_slices = [data[i:i+size] for i in ix]
        y_slices = [data[i+1:i+size+1] for i in ix]

        x = torch.stack(x_slices)
        y = torch.stack(y_slices)

        return x.to(self.device), y.to(self.device)

    def load_dataset(self):
        if self.dataset.exists():
            if self.dataset.is_dir():
                files = os.listdir(self.dataset)
                self.text = ''
                if len(files) > 0:
                    for file in files:
                        with open(self.dataset.joinpath(file), 'r') as f:
                            self.text += repr(f.read().replace('\\n', '\n'))
                else:
                    raise IOError('Dataset does not contain any files!')
            elif self.dataset.is_file():
                with open(self.dataset) as f:
                    self.text = repr(f.read().replace(
                        '\\n', '\n'))  # [:1_000_000] # Shortens training data for development.
        else:
            raise IOError('Dataset does not exist!')

    def load_tokenizer(self):
        if len(self.text) > 0:
            if self.version == 1:
                self.chars = sorted(list(set(self.text)))
                self.vocab_size = len(self.chars)
                # Create a mapping from characters to integers
                stoi = {ch: i for i, ch in enumerate(self.chars)}
                itos = {i: ch for i, ch in enumerate(self.chars)}
                # encoder: take a string, output a list of integers
                self.encode = lambda s: [stoi[c] for c in s]
                # decoder: take a list of integers, output a string
                self.decode = lambda l: ''.join([itos[i] for i in l])
            elif self.version == 2:
                self.tokenizer = Tokenizer_V1()
                self.tokenizer.load(self.text)
                self.vocab_size = len(self.tokenizer.tokens)
                print(f'Vocab size: {self.vocab_size}')
            elif self.version == 3:
                create_folder_if_not_exists('tmp')
                with open('tmp/vocab.txt', 'w+') as f:
                    f.write(self.text[:50_000_000].replace('\\n', '\n'))
                self.tokenizer = Tokenizer_V2()
                self.vocab_size = 4096
                self.tokenizer.train(
                    [Path('tmp/vocab.txt')], vocab_size=self.vocab_size)
            elif self.version == 4:
                self.tokenizer = Tokenizer_V3()
                self.tokenizer.load(self.text)
                self.vocab_size = len(self.tokenizer.tokens)
            print(f'Loaded {self.vocab_size} tokens.')
        else:
            raise Exception(
                'Could now load tokenizer due to lack of data in dataset.')

    def convert_dataset(self):
        print('Encoding...')
        if self.version == 1:
            self.data = torch.tensor(self.encode(self.text), dtype=torch.long)
        elif self.version == 2 or self.version == 4:
            self.data = torch.tensor(
                self.tokenizer.encode(self.text), dtype=torch.long)
        elif self.version == 3:
            self.data = self.encode_and_combine_chunks()
        n = int(1*len(self.data))  # first 90% will be train, rest val
        self.train_data = self.data[:n]
        self.val_data = self.data[n:]
        del (self.text)

    def create_model(self):
        if self.network == 'lstm':
            if self.version == 1:
                self.model = LFAI_LSTM(self.vocab_size, self.context_length,
                                       self.hiddensize, self.numlayers, self.device)
            else:
                self.model = LFAI_LSTM_V2(self.vocab_size, self.context_length,
                                          self.hiddensize, self.numlayers, self.device, half=self.half)
        elif self.network == 'gru':
            self.model = LFAI_GRU(self.vocab_size, self.context_length,
                                  self.hiddensize, self.numlayers, self.device, half=self.half)
        elif self.network == 'rnn':
            self.model = LFAI_RNN(self.vocab_size, self.context_length,
                                  self.hiddensize, self.numlayers, self.device, half=self.half)
        elif self.network == 'linear':
            self.model = LFAI_Linear(
                self.vocab_size, self.context_length, self.hiddensize, self.device, half=self.half)

        if len(self.load) > 0:
            data = torch.load(self.load, map_location=self.device)
            self.model.load_state_dict(data['state_dict'])

        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss().to(device=self.device)

        if self.half:
            self.model = self.model.half()
            self.criterion = self.criterion.half()

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate)

    def train(self):
        losses = []

        size = self.train_data.size(0)-1

        print('Processing full dataset...')

        for epoch in range(self.epochs):

            self.model.train()
            if self.network != 'linear':
                hidden = self.model.init_hidden(self.batch_size)
            td = tqdm(range(0, size, self.batch_size), dynamic_ncols=True)

            for _ in td:
                inputs_batch, targets_batch = self.get_batch('train')

                if self.half:
                    inputs_batch = inputs_batch.to(
                        torch.int16).to(torch.long)
                    targets_batch = targets_batch.to(
                        torch.int16).to(torch.long)

                self.optimizer.zero_grad()

                if self.network != 'linear':
                    outputs, hidden = self.model(inputs_batch, hidden)
                else:
                    outputs = self.model(inputs_batch)

                loss = self.criterion(
                    outputs.view(-1, self.vocab_size), targets_batch.view(-1))

                loss.backward()
                self.optimizer.step()

                if self.network != 'linear':
                    if self.network == 'lstm':
                        hidden = tuple(h.detach() for h in hidden)
                    else:
                        hidden = hidden.detach()

                if _ % (self.batch_size*2) == 0:
                    description = f'[ epoch: {epoch}, loss: {loss.item():.4f} ]'
                    td.set_description(description)
                    losses.append(loss.item())
                    if _ % (self.batch_size*6) == 0 and self.savelast:
                        self.save()
                    if _ % (self.batch_size*6) == 0 and self.graph:
                        create_folder_if_not_exists('graph')
                        plot_loss(losses[:50], 'graph/losses-'+self.name)
        self.save()

    def save(self, name: str = None):
        create_folder_if_not_exists('weights')
        if name == None:
            name = self.save_file
        if self.version == 1:
            save_out = {
                'vocab_size': self.vocab_size,
                'context_length': self.context_length,
                'hidden_size': self.hiddensize,
                'num_layers': self.numlayers,
                'chars': self.chars,
                'state_dict': self.model.state_dict(),
                'network': self.network,
                'version': self.version
            }
        elif self.version == 2 or self.version == 4:
            save_out = {
                'vocab_size': self.vocab_size,
                'context_length': self.context_length,
                'hidden_size': self.hiddensize,
                'num_layers': self.numlayers,
                'chars': self.tokenizer.tokens,
                'state_dict': self.model.state_dict(),
                'version': self.version,
                'network': self.network,
                'max_n_count': self.tokenizer.max_n_count
            }
        elif self.version == 3:
            save_out = {
                'vocab_size': self.vocab_size,
                'context_length': self.context_length,
                'hidden_size': self.hiddensize,
                'num_layers': self.numlayers,
                'chars': self.tokenizer.model.getvalue(),
                'state_dict': self.model.state_dict(),
                'network': self.network,
                'version': self.version
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
    parser.add_argument("--numlayers", default=4,
                        help="Specify how many layers the rnn layer will have.", required=False)
    parser.add_argument("--hiddensize", default=96,
                        help="Specify how large the hidden layer size will be.", required=False)
    parser.add_argument("--contextsize", default=96,
                        help="Specify the max length of the input context field.", required=False)
    parser.add_argument("--batchsize", default=16,
                        help="Specify how much data will be passed at one time.", required=False)
    parser.add_argument("--learningrate", default=0.001,
                        help="Specify how confident the model will be in itself.", required=False)
    parser.add_argument("--half", default=False, type=bool,
                        help="Specify if the model should use fp16 (Only for GPU).", required=False)
    parser.add_argument("--version", default=4,
                        help="Specify what version of the model.", required=False)
    parser.add_argument("--load", default="",
                        help="Specify a premade model to load.", required=False)
    parser.add_argument("--network", default="LSTM",
                        help="Specify a type of model (LSTM, GRU, RNN).", required=False)
    parser.add_argument("--graph", default=False, type=bool,
                        help="Specify if the model graph its losses.", required=False)
    parser.add_argument("--savelast", default=False, type=bool,
                        help="Specify if the model should save last or cosntant pth.", required=False)

    args = parser.parse_args()

    name = args.name
    dataset = args.dataset
    epochs = int(args.epochs)
    numlayers = int(args.numlayers)
    hiddensize = int(args.hiddensize)
    contextsize = int(args.contextsize)
    batch_size = int(args.batchsize)
    learningrate = float(args.learningrate)
    version = int(args.version)
    half = bool(args.half)
    load = str(args.load)
    network = str(args.network)
    graph = bool(args.graph)
    savelast = bool(args.savelast)

    trainer = Trainer(name, dataset, epochs,
                      contextsize, numlayers, hiddensize,
                      batch_size, learningrate, half,
                      version, load, network, graph, savelast)
    print('Loading dataset...')
    trainer.load_dataset()
    print('Loading tokenizer...')
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
