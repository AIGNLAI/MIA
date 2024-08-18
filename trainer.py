from model import ConvNet,ConvNet_for_MNIST,FCNet,ResNet18
from get_data import get_data
import copy
import numpy as np
import torchvision
import torch
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
from tqdm import tqdm
from sklearn.metrics import classification_report,accuracy_score
torch.multiprocessing.set_sharing_strategy('file_system')
MODEL_PATH = './attack_model/'

def full_attack_training(args):
    train_indices = list(range(args.TRAIN_EXAMPLES_AVAILABLE))
    train_target_indices = np.random.choice(train_indices, args.TRAIN_SIZE, replace=False)
    train_shadow_indices = np.setdiff1d(train_indices, train_target_indices)
    test_indices = list(range(args.TEST_EXAMPLES_AVAILABLE))
    test_target_indices = np.random.choice(test_indices, args.TEST_SIZE, replace=False)
    test_shadow_indices = np.setdiff1d(test_indices, test_target_indices)
    print("training target model...")
    attack_test_x, attack_test_y, test_classes = train_target_model(
        args,
        train_indices=train_target_indices,
        test_indices=test_target_indices,
        epochs=args.target_epochs,
        batch_size=args.target_batch_size,
        learning_rate=args.target_learning_rate,
        model=args.target_model,
        fc_dim_hidden=args.target_fc_dim_hidden,
        save=args.save_model)
    print("done training target model")

    print("training shadow models...")
    attack_train_x, attack_train_y, train_classes = train_shadow_models(
        args,
        train_indices=train_shadow_indices,
        test_indices=test_shadow_indices,
        epochs=args.target_epochs,
        batch_size=args.target_batch_size,
        learning_rate=args.target_learning_rate,
        n_shadow=args.n_shadow,
        fc_dim_hidden=args.target_fc_dim_hidden,
        model=args.target_model,
        save=args.save_model)
    print("done training shadow models")

    print("training attack model...")
    data = (attack_train_x, attack_train_y, train_classes,\
               attack_test_x, attack_test_y, test_classes)
    train_attack_model(
        args,
        data=data,
        epochs=args.attack_epochs,
        batch_size=args.attack_batch_size,
        learning_rate=args.attack_learning_rate,
        fc_dim_hidden=args.attack_fc_dim_hidden,
        model=args.attack_model)
    print("done training attack model")

def only_attack_training(args):
    dataset = None
    train_attack_model(
        args,
        dataset=dataset,
        epochs=args.attack_epochs,
        batch_size=args.attack_batch_size,
        learning_rate=args.attack_learning_rate,
        fc_dim_hidden=args.attack_fc_dim_hidden,
        model=args.attack_model)
    

def train_target_model(args, train_indices, test_indices,
                       epochs=100, batch_size=10, learning_rate=0.01,
                       fc_dim_hidden=50, model='cnn', save=True):
    trainset, testset = get_data(args)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=2,
                                              sampler=SubsetRandomSampler(train_indices),
                                              drop_last=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False,
                                             num_workers=2,
                                             sampler=SubsetRandomSampler(test_indices),
                                             drop_last=True)

    output_layer, _1, _2 = train(args, trainloader, testloader,
                                 fc_dim_hidden=fc_dim_hidden, epochs=epochs,
                                 learning_rate=learning_rate, batch_size=batch_size,
                                 model=model)
    attack_x, attack_y, classes = [], [], []

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        for data in tqdm(trainloader, desc="Processing train data"):
            images, labels = data[0].to(device), data[1].to(device)
            outputs = output_layer(images)
            attack_x.append(outputs.cpu())
            attack_y.append(np.ones(batch_size))
            classes.append(labels)
        for data in tqdm(testloader, desc="Processing test data"):
            images, labels = data[0].to(device), data[1].to(device)
            outputs = output_layer(images)
            attack_x.append(outputs.cpu())
            attack_y.append(np.zeros(batch_size))
            classes.append(labels)

    attack_x = np.vstack(attack_x)
    attack_y = np.concatenate(attack_y)
    classes = np.concatenate([cl.cpu() for cl in classes])

    attack_x = attack_x.astype('float32')
    attack_y = attack_y.astype('int32')
    classes = classes.astype('int32')

    if save:
        torch.save((attack_x, attack_y, classes), MODEL_PATH + 'attack_test_data.pth')

    return attack_x, attack_y, classes


def train_shadow_models(args, train_indices, test_indices,
                        fc_dim_hidden=50, n_shadow=20, model='cnn',
                        epochs=100, learning_rate=0.05, batch_size=10,
                        save=True):

    trainset, testset = get_data(args)
    attack_x, attack_y, classes = [], [], []

    for i in range(n_shadow):
        print(f'Training shadow model {i + 1}/{n_shadow}')
        
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=2,
                                                  sampler=SubsetRandomSampler(
                                                      np.random.choice(train_indices, args.TRAIN_SIZE,
                                                                       replace=False)),
                                                  drop_last=True)

        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=2,
                                                 sampler=SubsetRandomSampler(
                                                     np.random.choice(test_indices,
                                                                      round(args.TRAIN_SIZE * 0.3),
                                                                      replace=False)),
                                                 drop_last=True)

        output_layer, _1, _2 = train(args, trainloader, testloader,
                                     fc_dim_hidden=fc_dim_hidden, model=model,
                                     epochs=epochs, learning_rate=learning_rate,
                                     batch_size=batch_size)

        attack_i_x, attack_i_y, classes_i = [], [], []

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        with torch.no_grad():
            for data in tqdm(trainloader, desc=f"Processing train data for shadow model {i + 1}"):
                images, labels = data[0].to(device), data[1].to(device)
                outputs = output_layer(images)
                attack_i_x.append(outputs.cpu())
                attack_i_y.append(np.ones(batch_size))
                classes_i.append(labels)

            for data in tqdm(testloader, desc=f"Processing test data for shadow model {i + 1}"):
                images, labels = data[0].to(device), data[1].to(device)
                outputs = output_layer(images)
                attack_i_x.append(outputs.cpu())
                attack_i_y.append(np.zeros(batch_size))
                classes_i.append(labels)

        attack_x += attack_i_x
        attack_y += attack_i_y
        classes += classes_i

    attack_x = np.vstack(attack_x)
    attack_y = np.concatenate(attack_y)
    classes = np.concatenate([cl.cpu() for cl in classes])

    attack_x = attack_x.astype('float32')
    attack_y = attack_y.astype('int32')
    classes = classes.astype('int32')

    if save:
        torch.save((attack_x, attack_y, classes), MODEL_PATH + 'attack_test_data.pth')

    return attack_x, attack_y, classes


def reduce_ones(x, y, classes):
    idx_to_keep = np.where(y == 0)[0]
    idx_to_reduce = np.where(y == 1)[0]
    num_to_reduce = (y.shape[0] - idx_to_reduce.shape[0]) * 2
    idx_sample = np.random.choice(idx_to_reduce, num_to_reduce, replace = False)

    x = x[np.concatenate([idx_to_keep, idx_sample, idx_to_keep])]
    y = y[np.concatenate([idx_to_keep, idx_sample, idx_to_keep])]
    classes = classes[np.concatenate([idx_to_keep, idx_sample, idx_to_keep])]

    return x, y, classes

def train_attack_model(args,data = None,
                       fc_dim_hidden = 50, model = 'fc',
                       learning_rate = 0.01, batch_size = 10, epochs = 3):
    if data is None:
        data = load_attack_data()
    train_x, train_y, train_classes, test_x, test_y, test_classes = data

    train_x, train_y, train_classes = reduce_ones(train_x, train_y, train_classes)
    test_x, test_y, test_classes = reduce_ones(test_x, test_y, test_classes)

    train_indices = np.arange(len(train_x))
    test_indices = np.arange(len(test_x))
    unique_classes = np.unique(train_classes)
    true_y = []
    pred_y = []
    for c in unique_classes:
        print('Training attack model for class %d...'%(c))
        c_train_indices = train_indices[train_classes == c]
        c_train_x, c_train_y = train_x[c_train_indices], train_y[c_train_indices]
        c_test_indices = test_indices[test_classes == c]
        c_test_x, c_test_y = test_x[c_test_indices], test_y[c_test_indices]
        print("training number is %d"%c_train_x.shape[0])
        print("testing number is %d"%c_test_x.shape[0])

        trainloader = iterate_and_shuffle_numpy(c_train_x, c_train_y, batch_size)
        testloader = iterate_and_shuffle_numpy(c_test_x, c_test_y, batch_size)


        _, c_pred_y, c_true_y = train(args,trainloader, testloader,
                               fc_dim_in = train_x.shape[1],
                               fc_dim_out = 2,
                               fc_dim_hidden = fc_dim_hidden, epochs = epochs, learning_rate = learning_rate,
                               batch_size = batch_size, model = model)
        true_y.append(c_true_y)
        pred_y.append(c_pred_y)
        print("Accuracy score for class %d:"%c)
        print(accuracy_score(c_true_y, c_pred_y))

    true_y = np.concatenate(true_y)
    pred_y = np.concatenate(pred_y)
    print('Final full: %0.2f'%(accuracy_score(true_y, pred_y)))
    print(classification_report(true_y, pred_y))

def train(args, trainloader, testloader, model='cnn',
          fc_dim_hidden=50, fc_dim_in=10, fc_dim_out=2,
          batch_size=10, epochs=10,
          learning_rate=0.001):

    if model == 'fc':
        net = FCNet(dim_hidden=fc_dim_hidden, dim_in=args.classes, dim_out=fc_dim_out,
                    batch_size=batch_size)
    elif model == 'cnn' and args.d == 'MNIST':
        net = ConvNet_for_MNIST()
    elif model == 'cnn':
        net = ConvNet()
    elif model == 'resnet18':
        net = ResNet18(args.classes)
    else:
        raise NotImplementedError
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    
    criterion = nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

    bak_trainloader, bak_testloader = copy.deepcopy(trainloader), copy.deepcopy(testloader)
    needs_refresh = False
    if 'needs_refresh' in dir(trainloader):
        trainloader = bak_trainloader() 
        testloader = bak_testloader()
        needs_refresh = True
    
    for epoch in range(epochs):
        if needs_refresh:
            trainloader = bak_trainloader() 
            testloader = bak_testloader()
        
        running_loss = 0.0
        n_correct = 0
        n_total = 0

        for idx, data in enumerate(tqdm(trainloader, desc=f"Epoch {epoch + 1}/{epochs} - Training")):
            try:
                inputs, labels = data[0].to(device), data[1].to(device)
            except:
                inputs = torch.from_numpy(data[0]).to(device)
                labels = torch.from_numpy(data[1]).to(device)
            
            optimizer.zero_grad()
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            n_total += labels.size(0)
            n_correct += (predicted == labels).sum().item()
            
            try:
                loss = criterion(outputs, labels)
            except:
                loss = criterion(outputs, labels.long())
                
            loss.backward()
            optimizer.step()

        if epoch == epochs - 1:
            print(f'Epoch {epoch + 1}/{epochs} - Accuracy on training set: {100 * n_correct / n_total:.2f}%')
        
        n_correct, n_total = 0, 0
        y_hat, y_true = [], []
        
        with torch.no_grad():
            for idx, data in enumerate(tqdm(testloader, desc=f"Epoch {epoch + 1}/{epochs} - Testing")):
                try:
                    images, labels = data[0].to(device), data[1].to(device)
                except:
                    images = torch.from_numpy(data[0]).to(device)
                    labels = torch.from_numpy(data[1]).to(device)

                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                n_total += labels.size(0)
                n_correct += (predicted == labels).sum().item()
                
                if epoch == epochs - 1:
                    y_hat.append(predicted.cpu().numpy())
                    y_true.append(labels.cpu().numpy())

        if epoch == epochs - 1:
            print(f'Epoch {epoch + 1}/{epochs} - Accuracy on test set: {100 * n_correct / n_total:.2f}%')

    y_true, y_hat = np.concatenate(y_true), np.concatenate(y_hat)

    return net, y_hat, y_true


def iterate_and_shuffle_numpy(inputs, targets, batch_size):
    def return_generator():
        assert len(inputs) == len(targets)
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)

        for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
            excerpt = indices[start_idx:start_idx + batch_size]
            yield inputs[excerpt], targets[excerpt]

    return_generator.needs_refresh = True
    return return_generator
