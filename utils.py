import numpy as np


def generate_data_indices(data_size, target_train_size):
    #Returns indices for data sizing and sampling
    train_indices = np.arange(data_size)
    target_data_indices = np.random.choice(train_indices, target_train_size, replace = False)
    shadow_indices = np.setdiff1d(train_indices, target_data_indices)
    return target_data_indices, shadow_indices

def load_attack_data(MODEL_PATH):
    #Loads presaved training and testing datasets
    fname = MODEL_PATH + 'attack_train_data.pth'
    with np.load(fname) as f:
        train_x, train_y, train_classes = [f['arr_%d' % i]
                                           for i in range(len(f.files))]
    fname = MODEL_PATH + 'attack_test_data.pth'
    with np.load(fname) as f:
        test_x, test_y, test_classes = [f['arr_%d' % i]
                                        for i in range(len(f.files))]

    return train_x.astype('float32'),train_y.astype('int32'), train_classes.astype('int32'), test_x.astype('float32'), test_y.astype('int32'), test_classes.astype('int32'),


def get_more_args(args):
    if args.d == 'CIFAR10' or args.d == 'CIFAR100' or args.d == 'MNIST':
        args.TRAIN_SIZE = 10000
        args.TEST_SIZE = 500
        args.TRAIN_EXAMPLES_AVAILABLE = 50000
        args.TEST_EXAMPLES_AVAILABLE = 10000
