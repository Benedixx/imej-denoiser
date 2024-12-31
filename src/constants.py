import torch

DATA_PATH = 'data'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 1
EPOCHS = 50
N_EPOCHS_STOP = 5
IMAGE_SIZE = 256