from inference import Inference
from pathlib import Path
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='LFAI',
                        help="Specify a model path.", required=True)
    parser.add_argument("--prompt", default='LFAI',
                        help="Specify a prompt path.", required=True)
    args = parser.parse_args()

    inference = Inference(args.model)
    with open(Path(args.prompt), 'r') as f:
        output, hidden = inference.run(f.read(), ending_criterion='\n')
    print(output)
    while True:
        input_ = f'User: {input()}\nKamisatoAyaka:'
        output, hidden = inference.run(input_, hidden=hidden, ending_criterion='\n')
        print(output)