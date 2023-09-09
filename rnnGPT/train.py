import torch
import torch.nn as nn
from torch.nn import functional as F
from data_processing.tokenizer_generator import generate_tokenizer
from TRAIN_SETTINGS import *
import os

def load_dataset():
    with open(TEXT_DATASET, 'r', encoding='utf-8') as f:
        return f.read()

def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer_file = "./tokens.json"
    if os.path.exists(tokenizer_file):
        print(f"The path '{tokenizer_file}' exists.")
    else:
        print(f"The path '{tokenizer_file}' does not exist.")
        generate_tokenizer()
        print('Downloaded tokenizer chars.')

    

    # Train and test splits
    data = torch.tensor(encode(load_dataset()), dtype=torch.long)
    n = int(0.9*len(data)) # first 90% will be train, rest val
    train_data = data[:n]
    val_data = data[n:]
    
    


if __name__ == '__main__':
    train()