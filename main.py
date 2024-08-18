import numpy as np
import os
from trainer import *
from sklearn.metrics import *
import torch
from utils import *
from args import get_arg
import warnings
torch.multiprocessing.set_sharing_strategy('file_system')


if __name__ == '__main__':
    parser = get_arg()
    MODEL_PATH = './attack_model/'
    args = parser.parse_args()
    np.random.seed(args.s)
    DATA_PATH = f'./data/{args.d}'
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
    get_more_args(args)
    print(vars(args))
    if args.save_data:
        save_data()
    else:
        full_attack_training(args)


