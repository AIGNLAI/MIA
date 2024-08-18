import argparse

def get_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('-target_model', type=str,default="cnn",choices=['LeNet','resnet18'])
    parser.add_argument('-d', type=str,default="CIFAR10",choices=['CIFAR10','CIFAR100','MNIST'])
    parser.add_argument('-s', type=int,default=2024)
    parser.add_argument('--save_model', type = int, default = 1)
    parser.add_argument('--save_data', type = int, default = 0)
    parser.add_argument('--target_learning_rate', type = float, default = 0.001)
    parser.add_argument('--target_batch_size', type = int, default = 4)
    parser.add_argument('--target_fc_dim_hidden', type = int, default = 50)
    parser.add_argument('--target_epochs', type = int, default = 10)
    parser.add_argument('--n_shadow', type = int, default = 10)
    parser.add_argument('--attack_model', type = str, default = 'fc')
    parser.add_argument('--attack_learning_rate', type = float, default = 0.001)
    parser.add_argument('--attack_batch_size', type = int, default = 50)
    parser.add_argument('--attack_fc_dim_hidden', type = int, default = 50)
    parser.add_argument('--attack_epochs', type = int, default = 5)


    return parser