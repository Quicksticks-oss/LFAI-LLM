
# LFAI Documentation - Training

This is the official documentation for the LFAI lstm model.


# Training is easy!

## Youtube Tutorial
[![Youtube](https://img.youtube.com/vi/ZgcPPSaNNbQ/0.jpg)](https://www.youtube.com/watch?v=ZgcPPSaNNbQ)

## Dataset
The first step to start training your model is to find a suitable dataset for your model. A good place to find datasets for free is [Kaggle](https://kaggle.com/) in this example we will be going searching for datasets here [Kaggle Text Datasets](https://www.kaggle.com/search?q=text+datasetFileTypes%3Atxt).

In this example I will use the [Shakespeare text](https://www.kaggle.com/datasets/adarshpathak/shakespeare-text) because it is only 1M in size which is very small and perfect as a demo dataset.

I will now extract the dataset to the ``dataset`` folder in my git clone of LFAI.
My folder structure looks like.
```bash
dataset/
└── text.txt
0 directories, 1 file
```
Now that my data is downloaded we need to train a new model on that data.
Open a [Command Prompt](https://stackoverflow.com/questions/40146104/is-there-a-way-to-open-command-prompt-in-current-folder) or [Terminal](https://www.howtogeek.com/686955/how-to-launch-a-terminal-window-on-ubuntu-linux/) in the top level of the git clone/download of LFAI. 

## Training
We will now execute a command to start training on that text data!

### Windows 10/11
```bash
python train.py --name="Shakespeare" --dataset="dataset/text.txt" --batchsize=16 --contextsize=128
```
### Linux/Unix
```shell
python3 train.py --name="Shakespeare" --dataset="dataset/text.txt" --batchsize=16 --contextsize=128
```

# Final Chapter

This is the end you did it!
Now you can move on to the [Next Section](/docs/INFERENCE.md)!