
# LFAI Documentation - Advanced Training

This is the official documentation for the LFAI lstm model.

# More Coming Soon

Here are some arguments you can experiment with.

```shell
  --name NAME           Specify a model name.
  --epochs EPOCHS       Specify how many epochs the model should train with.
  --dataset DATASET     Specify the dataset the model will train on, this can be a file or
                        folder.
  --numlayers NUMLAYERS
                        Specify how many layers the rnn layer will have.
  --hiddensize HIDDENSIZE
                        Specify how large the hidden layer size will be.
  --contextsize CONTEXTSIZE
                        Specify the max length of the input context field.
  --batchsize BATCHSIZE
                        Specify how much data will be passed at one time.
  --learningrate LEARNINGRATE
                        Specify how confident the model will be in itself.
  --half HALF           Specify if the model should use fp16 (Only for GPU).
```