from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
import torchvision
def get_data(args):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    if args.d == 'CIFAR10':
        trainset=torchvision.datasets.CIFAR10(root='./data/CIFAR10', train = True, download = True,
                                            transform = transform)
        testset= torchvision.datasets.CIFAR10(root = './data/CIFAR10', train = False, download = True,
                                           transform = transform)
        args.classes = 10
    elif args.d == 'MNIST':
        trainset = torchvision.datasets.MNIST(root='./data/MNIST', train=True, download=True,
                                            transform=transform)
        testset = torchvision.datasets.MNIST(root='./data/MNIST', train=False, download=True,
                                           transform=transform)
        args.classes = 10
    elif args.d == "CIFAR100":
        trainset = torchvision.datasets.CIFAR100(root='./data/CIFAR100', train=True, download=True,
                                            transform=transform)
        testset = torchvision.datasets.CIFAR100(root='./data/CIFAR100', train=False, download=True,
                                           transform=transform)
        args.classes = 10
    return trainset,testset
    