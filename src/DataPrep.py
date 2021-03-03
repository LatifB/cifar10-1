import torch
import torchvision
import torchvision.transforms as transforms


class DataPrep:

    def __init__(self, batch_size=32):
        self.transform = transforms.Compose([transforms.RandomCrop(32, 5,
                                            padding_mode='reflect'),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.RandomRotation(10),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

        train_data = torchvision.datasets.CIFAR10(root='./data/cifar10-train', train=True,
                                                  download=True, transform=self.transform)
        test_data = torchvision.datasets.CIFAR10(root='./data/cifar10-test', train=False,
                                                 download=True, transform=self.transform)

        train_dataset, vald_dataset = torch.utils.data.random_split(train_data, [40000, 10000])
        __, mini_data = torch.utils.data.random_split(train_data, [45000, 5000])
        train_mini, vald_mini = torch.utils.data.random_split(mini_data, [4000, 1000])

        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.valid_loader = torch.utils.data.DataLoader(vald_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)
        self.train_mini_loader = torch.utils.data.DataLoader(train_mini, batch_size=batch_size, shuffle=True)
        self.vald_mini_loader = torch.utils.data.DataLoader(vald_mini, batch_size=batch_size, shuffle=True)

    def get_train_loader(self):
        return self.train_loader

    def get_valid_loader(self):
        return self.valid_loader

    def get_test_loader(self):
        return self.test_loader

    def get_train_mini_loader(self):
        return self.train_mini_loader

    def get_vald_mini_loader(self):
        return self.vald_mini_loader
