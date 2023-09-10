import torch
import torch.nn as nn
from torch.nn import functional as F
from tokenizer.tokenizer import CharBasedTokenizer
from TRAIN_SETTINGS import *
from model import LanguageModel
import json
import os

device = torch.device('cpu')
with open('tokens.json', 'r', encoding='utf-8') as f:
    chars = json.loads(f.read())['chars']

tokenizer = CharBasedTokenizer(chars)

# Load the state_dict into the model
model = LanguageModel(tokenizer.vocab_size, n_embd,
                          n_layer, CONTEXT_SIZE, device)
state_dict = torch.load("current.pt", map_location="cpu")
model.load_state_dict(state_dict)
model = model.to(device)

model.eval()
# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
genned, hidden = model.generate(context, max_new_tokens=2000)
genned = genned[0].tolist()
print(tokenizer.decode(genned))