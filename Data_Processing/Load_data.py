from torchvision.datasets import CIFAR10, EMNIST, FashionMNIST
from torchvision.datasets import ImageFolder

def Load_data(dataset, train_transform, test_transform=None, split_ratio=0.8):
    if test_transform is None:
        test_transform = train_transform

    if dataset == "cifar10":
        trainset = CIFAR10("data", train=True, download=True, transform=train_transform)
        testset = CIFAR10("data", train=False, download=True, transform=test_transform)
    
    elif dataset == "emnist":
        trainset = EMNIST("data", split="balanced", train=True, download=True, transform=train_transform)
        testset = EMNIST("data", split="balanced", train=False, download=True, transform=test_transform)
    
    elif dataset == "fmnist":
        trainset = FashionMNIST(root='data', train=True, download=True, transform=train_transform)
        testset = FashionMNIST(root='data', train=False, download=True, transform=test_transform)
        
    else:
        trainset = ImageFolder(root=f"data/{dataset}/train", transform=train_transform)
        testset = ImageFolder(root=f"data/{dataset}/test", transform=test_transform)
        
    return trainset, testset
    