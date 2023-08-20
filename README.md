
![Logo](images/banner.png)


# LFAI

This GitHub repository hosts an innovative project featuring an LSTM-based embedding GPT-like neural network. This network is designed to fuse diverse data modalities such as images, audio, sensor inputs, and text, creating a holistic and human-like sentient AI system with the ability to comprehend and respond across multiple data formats.

## Models

- [Shakespeare Small](https://huggingface.co/Quicksticks-oss/LFAI/blob/main/Shakespeare-0.8M-20230820-6-128-ctx128.pth)

## Screenshots

![Training](images/training.gif)
![Inference](images/inference.png)


## Usage/Examples

### Inference
```python
from inference import Inference

if __name__ == '__main__':
    inference = Inference('Model Path Here')
    output, hidden = inference.run('MENENIUS:')
    print(output)
```

### Training
```shell
clear && python3 train.py --name="Model Name Here" --dataset="Dataset File or Path here" --contextsize=128
```

## Roadmap

- Public Models

- Additional networks like GRU

- Use last token as new token in inference

- Add more integrations


## Documentation

[Documentation](docs/DOCUMENTATION.md)


## License

[Non-Share and Non-Modify License](LICENSE.MD)


## Authors

- [@QuickSticks-oss](https://github.com/Quicksticks-oss)

