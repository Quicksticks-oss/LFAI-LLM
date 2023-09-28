# LFAI-V2

## Training

All training settings can be set in the `TRAIN_SETTINGS.py` script.
The main settings you want to pay attention to are `TEXT_DATASET` and `max_iters`.

`TEXT_DATASET` is the file path that contains all of yor utf-8 or ascii text. This could be for example [Tiny Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt).

`max_iters` is how many itterations through the dataset you would like to run. I would reccoment setting this to a variable like `5000` if you have a lower teir GPU or CPU but if you have a high teir GPU I would set it to `25000` or `50000`.

## Finetuning

If you want to finetune a dataset all you need to do is set `FINETUNE` to `True` and set `LOAD_FILE` to the model you want to finetune.

## Inference

Just open the script `inference.py` and change the `MODEL` and `DEVICE` variable if needed and then run the script.
