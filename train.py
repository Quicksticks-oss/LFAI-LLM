#! /usr/bin/python3
from model.LFAI_LSTM import LFAI_LSTM
from model.LFAI_LSTM import LFAI_LSTM_V2
from pathlib import Path
from tqdm import tqdm
from tokenizer import *
from utils import *
import torch.nn as nn
import argparse
import datetime
import random
import torch
import time
import os

# Main trainer class.


class Trainer:
    def __init__(self, name, dataset, epochs, context_length, numlayers, hiddensize, batch_size=12, learning_rate=0.001, half: bool = False, version: int = 2) -> None:
        self.current_date = str(
            datetime.datetime.now().date()).replace('-', '')
        self.name = name
        self.dataset = Path(dataset)
        self.epochs = epochs
        self.version = version
        self.batch_size = batch_size
        self.numlayers = numlayers
        self.hiddensize = hiddensize
        self.learning_rate = learning_rate
        self.context_length = context_length
        self.half = half
        self.params = 0
        self.save_file = f'{name}-{self.params}M-{self.current_date}-{numlayers}-{hiddensize}-ctx{context_length}'
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
        self.save_file = f'{self.name}-{self.params}M-{self.current_date}-{self.numlayers}-{self.hiddensize}-ctx{self.context_length}'
        return self.params

    def get_batch(self, split:str='train'):
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
                    self.text = repr(f.read().replace('\\n', '\n'))[:500000]
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
            else:
                self.tokenizer = Tokenizer()
                self.tokenizer.load(self.text)
                self.vocab_size = len(self.tokenizer.tokens)
            print(f'Vocab size: {self.vocab_size}')
        else:
            raise Exception(
                'Could now load tokenizer due to lack of data in dataset.')

    def convert_dataset(self):
        if self.version == 1:
            self.data = torch.tensor(self.encode(self.text), dtype=torch.long)
        else:
            self.data = torch.tensor(
                self.tokenizer.encode(self.text), dtype=torch.long)
        n = int(1*len(self.data))  # first 90% will be train, rest val
        self.train_data = self.data[:n]
        self.val_data = self.data[n:]
        del (self.text)

    def create_model(self):
        if self.version == 1:
            self.model = LFAI_LSTM(self.vocab_size, self.context_length,
                                   self.hiddensize, self.numlayers, self.device)
        else:
            self.model = LFAI_LSTM_V2(self.vocab_size, self.context_length,
                                      self.hiddensize, self.numlayers, self.device)
        if self.half:
            self.model.half()
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate)

    def train(self):
        '''
        This is the main training script.
         - First step: Learn basic language.
         - Final step: Cover whole dataset.
        '''
        total_steps = int((self.epochs+2)/2)
        for ep in range(2):

            if ep == 0:
                if self.train_data.size(0)-1 < 16_000:
                    size = ((self.train_data.size(0)-1) - self.context_length) - \
                        (self.context_length*self.batch_size)
                else:
                    size = 16_000
                print('Processing partial dataset...')
            else:
                size = ((self.train_data.size(0)-1) - self.context_length) - \
                    (self.context_length*self.batch_size)
                print('Processing full dataset...')

            for epoch in range(total_steps):
                self.model.train()
                hidden = self.model.init_hidden(self.batch_size)

                td = tqdm(range(0, size),
                          postfix='training in progress...', dynamic_ncols=True)

                for _ in td:
                    inputs_batch, targets_batch = self.get_batch('train')

                    if self.half:
                        inputs_batch = inputs_batch.to(torch.int16)
                        targets_batch = targets_batch.to(torch.int16)

                    self.optimizer.zero_grad()
                    outputs, hidden = self.model(inputs_batch, hidden)
                    loss = self.criterion(
                        outputs.view(-1, self.vocab_size), targets_batch.view(-1))
                    loss.backward()
                    self.optimizer.step()
                    hidden = tuple(h.detach() for h in hidden)

                    if _ % 8 == 0:
                        description = f'[ epoch: {epoch}, loss: {loss.item():.4f} ]'
                        td.set_description(description)
                        if _ % 256 == 0:
                            self.save()

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
                'version': self.version
            }
        else:
            save_out = {
                'vocab_size': self.vocab_size,
                'context_length': self.context_length,
                'hidden_size': self.hiddensize,
                'num_layers': self.numlayers,
                'chars': self.tokenizer.tokens,
                'state_dict': self.model.state_dict(),
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
    parser.add_argument("--numlayers", default=6,
                        help="Specify how many layers the rnn layer will have.", required=False)
    parser.add_argument("--hiddensize", default=128,
                        help="Specify how large the hidden layer size will be.", required=False)
    parser.add_argument("--contextsize", default=256,
                        help="Specify the max length of the input context field.", required=False)
    parser.add_argument("--batchsize", default=16,
                        help="Specify how much data will be passed at one time.", required=False)
    parser.add_argument("--learningrate", default=0.001,
                        help="Specify how confident the model will be in itself.", required=False)
    parser.add_argument("--half", default=False,
                        help="Specify if the model should use fp16 (Only for GPU).", required=False)
    parser.add_argument("--version", default=2,
                        help="Specify what version of the model.", required=False)

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

    trainer = Trainer(name, dataset, epochs,
                      contextsize, numlayers, hiddensize,
                      batch_size, learningrate, half, version)
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
