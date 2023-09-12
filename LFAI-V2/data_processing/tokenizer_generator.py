import requests
import base64
import json

def generate_tokenizer(_type='json'):
    print('Downloading tokens...')
    text = requests.get("https://en.wikipedia.org/wiki/List_of_Unicode_characters").text

    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    # create a mapping from characters to integers
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
    decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

    if _type == 'json':
        j_data = json.dumps({'chars':chars}, ensure_ascii=False)
        with open('tokens.json', 'w+', encoding='utf-8') as f:
            f.write(j_data)
    elif _type == 'base64':
        with open('tokens.b64', 'w+', encoding='utf-8') as f:
            f.write(base64.b64encode(j_data.encode()).decode())
    print('Done!')

if __name__ == '__main__':
    generate_tokenizer()