'''
Based off of karpathy's nanoGPT
'''
import torch
import torch.nn as nn
from torch.nn import functional as F

class LanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embd, n_layer, context_size, device):
        super(LanguageModel, self).__init__()
        self.vocab_size = vocab_size
        self.context_size = context_size
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.device = device
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(context_size, n_embd)
        self.blocks = nn.LSTM(n_embd, n_embd, n_layer, batch_first=True)
        # Combine LayerNorm and Linear layers
        self.lm_head = nn.Sequential(
            nn.LayerNorm(n_embd),
            nn.Linear(n_embd, vocab_size)
        )

    def forward(self, idx, targets=None, hidden=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)

        if hidden == None:
            x, hidden = self.blocks(x) # (B,T,C)
        else:
            x, hidden = self.blocks(x, hidden) # (B,T,C)
        
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

        return logits, hidden, loss

    def generate(self, idx, max_new_tokens, hidden=None):
        idx_export = idx
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last context_size tokens
            idx_cond = idx[:, -self.context_size:]
            # get the predictions
            logits, hidden, loss = self(idx_cond, hidden=hidden)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = idx_next
            idx_export = torch.cat((idx_export, idx_next), dim=1) # (B, T+1)
        return idx_export, hidden