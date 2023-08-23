import re
from collections import Counter

class Tokenizer_V3:
    def __init__(self) -> None:
        self.tokens = {}
        self.token_pattern = re.compile(r'\b\w+\b|[.,!?;:]|[ \t\n\r\f\v]')
        self.max_n_count = 1

    def _find_words(self ,text:str):
        words = self.token_pattern.findall(repr(text).replace('\\n', '\n'))
        tokens = []
        for word in words:
            if len(word) > self.max_n_count:
                chunks = [word[i:i+self.max_n_count] for i in range(0, len(word), self.max_n_count)]
                tokens.extend(chunks)
            else:
                tokens.append(word)
        return tokens

    def load(self, text: str) -> None:
        token_counts = Counter(self._find_words(text))
        self.tokens['\\'] = 0
        for index, (token, _) in enumerate(token_counts.most_common()):
            self.tokens[token] = index+1

    def encode(self, text: str) -> str:
        tokens = self._find_words(text)
        encoded_text = [self.tokens[token] for token in tokens]
        return encoded_text

    def decode(self, encoded_text: str) -> str:
        reversed_tokens = {v: k for k, v in self.tokens.items()}
        decoded_tokens = [reversed_tokens[token] for token in encoded_text]
        
        decoded_text = ""
        for token in decoded_tokens:
            if len(token) > self.max_n_count:
                chunks = [token[i:i+self.max_n_count] for i in range(0, len(token), self.max_n_count)]
                decoded_text += "".join(chunks)
            else:
                decoded_text += token
        
        return decoded_text


if __name__ == '__main__':
    # Example usage
    input_text = """Hello, world! This is a basic super basic sub word tokenizer.
test 123"""
    tokenizer = Tokenizer_V3()
    tokenizer.load(input_text)

    tokenize = '''Hello world basic sub tokenizer.
12'''
    encoded = tokenizer.encode(tokenize)
    decoded = tokenizer.decode(encoded)
    print(len(encoded), len(tokenize))
    print(f'Vocab Size: {len(tokenizer.tokens)}')
    print(f'Encoded: {repr(encoded)}')
    print(f'Decoded: {repr(decoded)}')
