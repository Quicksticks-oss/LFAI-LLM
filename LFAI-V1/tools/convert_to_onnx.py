import torch.onnx
import sys
sys.path.append("../model")
from LFAI_LSTM import LFAI_LSTM
from LFAI_LSTM import LFAI_LSTM_V2
import argparse
import json


def main(model_path, version):
    # Example input tensor (adjust the shape and data type according to your model)

    data = torch.load(model_path, map_location='cpu')
    chars = data['chars']
    context_size = data['context_length']

    if version == 2:
        model = LFAI_LSTM_V2(
            data['vocab_size'], data['context_length'], data['hidden_size'], data['num_layers'], device='cpu', dropout_p=0.9)
    else:
        model = LFAI_LSTM(
            data['vocab_size'], data['context_length'], data['hidden_size'], data['num_layers'], device='cpu', dropout_p=0.9)
        
    with open(model_path.replace('.pth','.json'), 'w+') as f:
        x = {
            'vocab_size': data['vocab_size'],
            'context_length': data['context_length'],
            'hidden_size': data['hidden_size'],
            'num_layers': data['num_layers'],
            'chars': data['chars'],
            'version': data['version'],
            'network': data['network'],
            'max_n_count': data['max_n_count']
        }
        f.write(json.dumps(x))

    hidden = model.init_hidden(1, inference=True)

    dummy_input = torch.randint(high=data['vocab_size'], low=0, size=(1,1))

    # Export the model to ONNX format
    onnx_filename = model_path.replace('.pth','.onnx')
    torch.onnx.export(model, dummy_input, onnx_filename, verbose=False)
    print(f'Exported onnx model to {onnx_filename}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='LFAI',
                        help="Specify a model path.", required=True)
    parser.add_argument("--version", default=2,
                        help="Specify a model path.", required=True)
    args = parser.parse_args()
    main(args.model, int(args.version))
