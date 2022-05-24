import torch

# define training hyper parameters
INIT_LR = 1e-3
BATCH_SIZE = 64
EPOCHS = 10

# DEFINE the train and val splits
TRAIN_SPLIT = 0.75
VAL_SPLIT = 1 - TRAIN_SPLIT

# device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
