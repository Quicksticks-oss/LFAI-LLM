import sentencepiece as spm
import io

class Tokenizer_V2():
    def __init__(self) -> None:
        self.sp = None
        self.model = None

    def train(self, files:list, vocab_size:int=16000):
        # Train the SentencePiece tokenizer
        self.model = io.BytesIO()
        spm.SentencePieceTrainer.train(input=files, model_writer=self.model, vocab_size=vocab_size, user_defined_symbols=['START', 'END'])
        self.load(self.model.getvalue())
        return self.model.getvalue()

    def load(self, model):
        self.sp = spm.SentencePieceProcessor(model_proto=model)

    def encode(self, text:str):
        return self.sp.encode(text)
    
    def decode(self, tokens):
        return self.sp.decode(tokens)
