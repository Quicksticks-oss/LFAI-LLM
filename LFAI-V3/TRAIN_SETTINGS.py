# Variables

NAME = "My Model"

TEXT_DATASET = 'input.txt'
CONTEXT_SIZE = 128  # what is the maximum context length for predictions?

FINETUNE = False # Enable this to finetune
LOAD_FILE = "model_new.pt"

SAVE_FILE = "final.pt"

batch_size = 32

max_iters = 500
eval_interval = 250
eval_iters = 100

learning_rate = 1e-2

NEMBD = 512 # Best suited for 1gb-15gb datasets 14m params
NLAYER = 4
# -----------
