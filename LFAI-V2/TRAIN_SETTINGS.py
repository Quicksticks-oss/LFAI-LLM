# Variables

TEXT_DATASET = 'dataset.txt'
CONTEXT_SIZE = 512  # what is the maximum context length for predictions?

FINETUNE = False # Enable this to finetune
LOAD_FILE = "model_new.pt"

SAVE_FILE = "final.pt"

batch_size = 32

max_iters = 25000
eval_interval = 25
eval_iters = 200

learning_rate = 1e-2

NEMBD = 256
NLAYER = 6
# -----------