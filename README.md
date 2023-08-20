
![Logo](images/banner.png)


# LFAI

This GitHub repository hosts an innovative project featuring an LSTM-based embedding GPT-like neural network. This network is designed to fuse diverse data modalities such as images, audio, sensor inputs, and text, creating a holistic and human-like sentient AI system with the ability to comprehend and respond across multiple data formats.

## Screenshots

![Training](images/training.gif)
![Inference](images/training.gif)


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
clear && python3 train.py --name="Model Name Here" --dataset="Dataset File or Path here" --batchsize=32 --contextsize=128 --epochs=2
```

## Roadmap

- Public Models

- Additional networks like GRU

- Use last token as new token in inference

- Add more integrations


## Documentation

[Documentation](https://linktodocumentation)


## License

[Non-Share and Non-Modify License](LICENSE.MD)


## Authors

- [@QuickSticks-oss](https://github.com/Quicksticks-oss)

