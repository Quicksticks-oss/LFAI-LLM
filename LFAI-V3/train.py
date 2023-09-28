import torch
import torch.nn as nn
from torch.nn import functional as F
from data_processing.tokenizer_generator import generate_tokenizer
from tokenizer.tokenizer import CharBasedTokenizer
from TRAIN_SETTINGS import *
import TRAIN_SETTINGS
from model import LanguageModel
import json
from tqdm import tqdm
import os


def load_dataset():
    with open(TEXT_DATASET, 'r', encoding='utf-8') as f:
        return f.read()


def load_chars():
    with open('tokens.json', 'r', encoding='utf-8') as f:
        return json.loads(f.read())['chars']

# data loading


def get_batch(data, device):
    # generate a small batch of data of inputs x and targets y
    ix = torch.randint(len(data) - CONTEXT_SIZE, (batch_size,))
    x = torch.stack([data[i:i+CONTEXT_SIZE] for i in ix])
    y = torch.stack([data[i+1:i+CONTEXT_SIZE+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model, data, device):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(data, device)
            logits, hidden, loss = model(X, Y)
            losses[k] = loss.item()
            hidden = tuple(h.detach() for h in hidden)
        out[split] = losses.mean()
    model.train()
    return out


def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    tokenizer_file = "./tokens.json"
    if os.path.exists(tokenizer_file):
        print(f"The path '{tokenizer_file}' exists.")
    else:
        print(f"The path '{tokenizer_file}' does not exist.")
        generate_tokenizer()
        print('Downloaded tokenizer chars.')
    print('Loading tokenizer...')
    chars = load_chars()
    tokenizer = CharBasedTokenizer(chars)
    del (chars)

    print('Tokenizing training data...')
    # Train and test splits
    data = torch.tensor(tokenizer.encode(load_dataset()), dtype=torch.long)
    n = int(0.9*len(data))  # first 90% will be train, rest val
    train_data = data[:n]
    val_data = data[n:]
    print('Creating Neural Network...')
    if not FINETUNE:
        model = LanguageModel(tokenizer.vocab_size, TRAIN_SETTINGS.NEMBD,
                            TRAIN_SETTINGS.NLAYER, TRAIN_SETTINGS.CONTEXT_SIZE, device)
    else:
        print('Currently loading model to finetune.')
        save_out = torch.load(LOAD_FILE)
        n_embd = save_out["n_embd"]
        n_layer = save_out["n_layer"]
        ctx = save_out["ctx"]
        model = LanguageModel(tokenizer.vocab_size, n_embd,
                            n_layer, ctx, device)
        model.load_state_dict(save_out["out"], strict=False)
    model = model.to(device)
    # print the number of parameters in the model
    print(
        f'Network contains {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters...')
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    td = tqdm(range(0, max_iters), dynamic_ncols=True)

    for iter in td:

        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(model, val_data, device)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            save_out = {
                "out": model.state_dict(),
                "ctx": TRAIN_SETTINGS.CONTEXT_SIZE,
                "n_embd": TRAIN_SETTINGS.NEMBD,
                "n_layer": TRAIN_SETTINGS.NLAYER,
                "chars": tokenizer.chars
            }
            torch.save(save_out, 'current.pt')

        # sample a batch of data
        xb, yb = get_batch(train_data, device)

        # evaluate the loss
        _, hidden, loss = model(xb, yb)
        hidden = tuple(h.detach() for h in hidden)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        description = f'{TRAIN_SETTINGS.NAME} - [ epoch: [{iter}/{max_iters}], loss: {loss.item():.4f} ]'
        td.set_description(description)

    save_out = {
        "out": model.state_dict(),
        "ctx": TRAIN_SETTINGS.CONTEXT_SIZE,
        "n_embd": TRAIN_SETTINGS.NEMBD,
        "n_layer": TRAIN_SETTINGS.NLAYER,
        "chars": tokenizer.chars
    }
    torch.save(save_out, SAVE_FILE)

    # generate from the model
    context = torch.tensor([tokenizer.encode("User: ")], dtype=torch.long, device=device) # torch.zeros((1, 1), dtype=torch.long, device=device)
    genned, hidden = model.generate(context, max_new_tokens=2000)
    genned = genned[0].tolist()
    print(tokenizer.decode(genned))


if __name__ == '__main__':
    train()
