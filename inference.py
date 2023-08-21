#! /usr/bin/python3
from model.LFAI_LSTM import LFAI_LSTM
from model.LFAI_LSTM import LFAI_LSTM_V2
from pathlib import Path
from tokenizer import *
from tqdm import tqdm
from utils import *
import torch.nn as nn
import argparse
import datetime
import random
import torch
import os

# Main inference class.


class Inference:
    def __init__(self, model_path: str, version:int) -> None:
        self.version = version
        self.model_path = Path(model_path)
        self.load_model()
        self.load_tokenizer()

    def load_model(self):
        data = torch.load(self.model_path, map_location='cpu')
        self.chars = data['chars']
        if self.version == 1:
            self.model = LFAI_LSTM(
                data['vocab_size'], data['context_length'], data['hidden_size'], data['num_layers'])
        else:
            self.model = LFAI_LSTM_V2(
                data['vocab_size'], data['context_length'], data['hidden_size'], data['num_layers'], device='cpu')
        self.model.load_state_dict(data['state_dict'])
        self.model.eval()

    def load_tokenizer(self):
        if self.version == 1:
            # Create a mapping from characters to integers
            stoi = {ch: i for i, ch in enumerate(self.chars)}
            itos = {i: ch for i, ch in enumerate(self.chars)}
            # encoder: take a string, output a list of integers
            self.encode = lambda s: [stoi[c] for c in s]
            # decoder: take a list of integers, output a string
            self.decode = lambda l: ''.join([itos[i] for i in l])
        else:
            self.tokenizer = Tokenizer()
            self.tokenizer.tokens = self.chars
            self.vocab_size = len(self.tokenizer.tokens)

    def run(self, input_data:str):
        hidden = self.model.init_hidden(1)

        with torch.no_grad():
            if self.version == 1:
                input_sequence = torch.tensor(self.encode(input_data), dtype=torch.long).unsqueeze(0)
            else:
                input_sequence = torch.tensor(self.tokenizer.encode(input_data), dtype=torch.long).unsqueeze(0)
            # Initialize the output sequence with the input sequence
            output_sequence = input_sequence

            # Generate the rest of the sequence
            for _ in range(32):
                output, hidden = self.model(input_sequence, hidden)
                output = output[:, -1, :]
                _, topk = torch.topk(output, 1)
                input_sequence = topk.squeeze(0).unsqueeze(0)
                output_sequence = torch.cat((output_sequence, input_sequence), dim=1)

            if self.version == 1:
                generated_text = self.decode(output_sequence.squeeze().tolist())
            else:
                generated_text = self.tokenizer.decode(output_sequence.squeeze().tolist())
            return generated_text, hidden


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='LFAI',
                        help="Specify a model path.", required=True)
    parser.add_argument("--prompt", default='LFAI',
                        help="Specify a prompt path.", required=True)
    parser.add_argument("--version", default=2,
                        help="Specify the model's version.", required=False)
    args = parser.parse_args()

    inference = Inference(args.model, int(args.version))
    with open(Path(args.prompt), 'r') as f:
        output, hidden = inference.run(f.read().replace('\n','\\n'))
        print(output.replace('\\n','\n'))
