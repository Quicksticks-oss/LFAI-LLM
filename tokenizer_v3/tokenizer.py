import re
from collections import Counter


class Tokenizer:
    def __init__(self) -> None:
        self.tokens = {}
        self.token_pattern = re.compile(r'\b\w+\b|[.,!?;]|[ \t\n\r\f\v]')

    def load(self, text: str, maxtokenlength:int=3) -> None:
        self.mtl = maxtokenlength
        self.tokens[self.mtl] = self.mtl
        words = self.token_pattern.findall(text)
        tokens = [word[i:i+3] for word in words if len(word) > 3 for i in range(0, len(word), 3)] + [word for word in words if len(word) <= 3]

        token_counts = Counter(tokens)
        for index, (token, _) in enumerate(token_counts.most_common()):
            self.tokens[token] = index

    def encode(self, text: str) -> str:
        words = self.token_pattern.findall(text)
        tokens = []
        for word in words:
            if len(word) > 3:
                chunks = [word[i:i+3] for i in range(0, len(word), 3)]
                tokens.extend(chunks)
            else:
                tokens.append(word)
        encoded_text = [self.tokens[token] for token in tokens]
        return encoded_text

    def decode(self, encoded_text: str) -> str:
        reversed_tokens = {v: k for k, v in self.tokens.items()}
        decoded_tokens = [reversed_tokens[token] for token in encoded_text]
        
        decoded_text = ""
        for token in decoded_tokens:
            if len(token) > 3:
                chunks = [token[i:i+3] for i in range(0, len(token), 3)]
                decoded_text += "".join(chunks)
            else:
                decoded_text += token
        
        return decoded_text


if __name__ == '__main__':
    # Example usage
    input_text = """Hello, world! This is a basic super basic tokenizer."""
    tokenizer = Tokenizer()
    tokenizer.load(input_text)

    tokenize = 'Hello world basic tokenizer.'
    encoded = tokenizer.encode(tokenize)
    decoded = tokenizer.decode(encoded)

    print(len(encoded), len(tokenize))

    print(f'Vocab Size: {tokenizer.tokens}')
    print(f'Encoded: {encoded}')
    print(f'Decoded: {decoded}')
