'''
Based off of karpathy's nanoGPT
'''


class CharBasedTokenizer:
    def __init__(self, chars) -> None:
        self.chars = chars
        self.vocab_size = len(self.chars)
        # create a mapping from characters to integers
        stoi = {ch: i for i, ch in enumerate(self.chars)}
        itos = {i: ch for i, ch in enumerate(self.chars)}
        # encoder: take a string, output a list of integers
        self.encode = lambda s: [stoi.get(c, 0) for c in s]
        # decoder: take a list of integers, output a string
        self.decode = lambda l: ''.join([itos[i] for i in l])
