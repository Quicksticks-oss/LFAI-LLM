
# LFAI Inference

This is the official documentation for the LFAI lstm model.

# Model & Prompt

We first need to find our models path which is be default stored in the ``weights`` folder. The file that has ``Shakespeare`` in the title is what we named our model in the other example about training is our models path.

There is a file on the top level called ``prompt.txt`` this contains out prompt which out model will use as input. Please edit this file as you see fit!

# Inference
Inference is very easy to use.

To run inference we just need to execute the following command.

### Windows 10/11
```bash
python inference.py --name="weights/our full models name" --prompt="prompt.txt"
```
### Linux/Unix
```shell
python3 inference.py --name="weights/our full models name" --prompt="prompt.txt"
```

# Final Chapter

This is the end you did it!
Back to [Docs](/docs/DOCUMENTATION.md)!