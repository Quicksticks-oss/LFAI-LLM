# Variables

NAME = "My Model"

TEXT_DATASET = 'dataset.txt'
CONTEXT_SIZE = 512  # what is the maximum context length for predictions?

FINETUNE = False # Enable this to finetune
LOAD_FILE = "model_new.pt"

SAVE_FILE = "final.pt"

batch_size = 32

max_iters = 5000
eval_interval = 25
eval_iters = 200

learning_rate = 1e-2

NEMBD = 1024 # Best suited for 1gb-5gb datasets
NLAYER = 6
# -----------
