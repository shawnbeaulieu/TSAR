import torchvision.transforms as transforms
import torchvision

import datasets.omniglot as om

class DatasetFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_dataset(name, train=True, path=None, background=True, all=False):

        if name == "omniglot":
            train_transform = transforms.Compose(
                [transforms.Resize((84, 84)),
                 transforms.ToTensor()])
            if path is None:
                return om.Omniglot("../data/omni", background=background, download=True, train=train,
                                   transform=train_transform, all=all)
            else:
                return om.Omniglot(path, download=True, background=train, transform=train_transform)


        elif name == "CIFAR100":
                transform = transforms.Compose([
                                    transforms.Resize((28, 28)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
                ])

                return torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    
        else:
            print("Unsupported Dataset")
            assert False
