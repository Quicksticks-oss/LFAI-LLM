#! /usr/bin/python3
from model.LFAI_LSTM import LFAI_LSTM
from model.LFAI_LSTM import LFAI_LSTM_V2
from model.LFAI_GRU import LFAI_GRU
from model.LFAI_RNN import LFAI_RNN
from tokenizers.tokenizer_v3 import Tokenizer_V3
from tokenizers.tokenizer_v2 import Tokenizer_V2
from tokenizers.tokenizer_v1 import Tokenizer_V1
from pathlib import Path
import torch.nn as nn
import argparse
import torch

# Main inference class.


class Inference:
    def __init__(self, model_path: str) -> None:
        self.version = 2
        self.model_path = Path(model_path)
        self.load_model()
        self.load_tokenizer()

    def load_model(self):
        data = torch.load(self.model_path, map_location='cpu')
        self.chars = data['chars']
        self.context_size = data['context_length']
        self.vocab_size = data['vocab_size']
        self.version = int(data['version'])
        self.network = data['network']
        if self.network == 'lstm':
            if self.version == 1:
                self.model = LFAI_LSTM(
                    data['vocab_size'], data['context_length'], data['hidden_size'], data['num_layers'])
            else:
                self.model = LFAI_LSTM_V2(
                    data['vocab_size'], data['context_length'], data['hidden_size'], data['num_layers'], device='cpu', dropout_p=0.9)
        elif self.network == 'gru':
            self.model = LFAI_GRU(self.vocab_size, self.context_size,
                        data['hidden_size'], data['num_layers'], device='cpu', dropout_p=0.9)
        elif self.network == 'rnn':    
            self.model = LFAI_RNN(self.vocab_size, self.context_size,
                        data['hidden_size'], data['num_layers'], device='cpu', dropout_p=0.9)
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
        elif self.version == 2:
            self.tokenizer = Tokenizer_V1()
            self.tokenizer.tokens = self.chars
            # = len(self.tokenizer.tokens)
        elif self.version == 3:
            self.tokenizer = Tokenizer_V2()
            self.tokenizer.load(self.chars)
        elif self.version == 4:
            self.tokenizer = Tokenizer_V3()
            self.tokenizer.tokens = self.chars

    def run(self, input_data: str, ending_criterion:str=None):
        hidden = self.model.init_hidden(1, inference=True)
        if ending_criterion != None:
            ending_criterion = self.tokenizer.encode(ending_criterion)[0]

        with torch.no_grad():
            if self.version == 1:
                input_sequence = torch.tensor(self.encode(
                    input_data), dtype=torch.long).unsqueeze(0)
            else:
                input_sequence = torch.tensor(self.tokenizer.encode(
                    input_data), dtype=torch.long).unsqueeze(0)
                
            print(input_sequence.shape)
                
            # Initialize the output sequence with the input sequence
            output_sequence = input_sequence

            # Generate the rest of the sequence
            for _ in range(self.context_size):
                output, hidden = self.model(input_sequence, hidden)
                output = output[:, -1, :]
                _, topk = torch.topk(output, 1)
                input_sequence = topk.squeeze(0).unsqueeze(0)
                output_sequence = torch.cat((output_sequence, input_sequence), dim=1)
                if input_sequence.item() == ending_criterion:
                    break

            if self.version == 1:
                generated_text = self.decode(
                    output_sequence.squeeze().tolist())
            else:
                generated_text = self.tokenizer.decode(
                    output_sequence.squeeze().tolist())
            return generated_text, hidden


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='LFAI',
                        help="Specify a model path.", required=True)
    parser.add_argument("--prompt", default='LFAI',
                        help="Specify a prompt path.", required=True)
    parser.add_argument("--endingcriterion", default=None,
                        help="Specify a ending criterion string.", required=False)
    args = parser.parse_args()

    inference = Inference(args.model)
    with open(Path(args.prompt), 'r') as f:
        output, hidden = inference.run(f.read().replace('\n', '\\n'), args.endingcriterion)
        print(output.replace('\\n', '\n'))
