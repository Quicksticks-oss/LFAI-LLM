import torch
import torch.nn as nn
from torch.nn import functional as F
from tokenizer.tokenizer import CharBasedTokenizer
from model import LanguageModel

### PARAMS ###
MODEL = "LFAI-books-ctx512-2m.pth"
DEVICE = 'cpu'
### ###### ###

if __name__ == '__main__':
    device = torch.device(DEVICE)

    save_out = torch.load(MODEL, map_location="cpu")
    n_embd = save_out["n_embd"]
    n_layer = save_out["n_layer"]
    CONTEXT_SIZE = save_out["ctx"]
    chars = save_out['chars']

    tokenizer = CharBasedTokenizer(chars)

    # Load the state_dict into the model
    model = LanguageModel(tokenizer.vocab_size, n_embd,
                            n_layer, CONTEXT_SIZE, device)
    model.load_state_dict(save_out["out"])
    model = model.to(device)
    model.eval()

    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    genned, hidden = model.generate(context, max_new_tokens=2000)
    genned = genned[0].tolist()
    print(tokenizer.decode(genned))