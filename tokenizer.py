import re

class Tokenizer:
    def __init__(self) -> None:
        self.tokens = []

    def load(self, text:str) -> None:
        token_pattern = re.compile(r'\b\w+\b|[.,!?;]|[ \t\n\r\f\v]')
        self.tokens = sorted(set(token_pattern.findall(text)))

    def encode(self, text:str) -> str:
        tokens = re.findall(r'\b\w+\b|[.,!?;]|[ \t\n\r\f\v]', text)
        encoded_text = [self.tokens.index(token) for token in tokens]
        return encoded_text

    def decode(self, encoded_text:list):
        decoded_tokens = [self.tokens[index] for index in encoded_text]
        decoded_text = ''.join(decoded_tokens)
        return decoded_text

if __name__ == '__main__':
    # Example usage
    input_text = """Hello, world! This is a basic super basic tokenizer."""
    tokenizer = Tokenizer()
    tokenizer.load(input_text)
    encoded = tokenizer.encode('Hello basic tokenizer.')
    decoded = tokenizer.decode(encoded)

    print(tokenizer.tokens)
    print(encoded)
    print(decoded)
