import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        torch.manual_seed(14)
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer',
                        'dog', 'frog', 'horse', 'ship', 'truck')
        # conv layers
        self.conv1 = nn.Conv2d(3, 48, 3, padding=1)
        self.conv2 = nn.Conv2d(48, 96, 3, padding=1)
        self.conv3 = nn.Conv2d(96, 192, 3, padding=1)
        self.conv4 = nn.Conv2d(192, 256, 3, padding=1)
        # maxpool and drop
        self.pool2 = nn.MaxPool2d(2)
        self.drop = nn.Dropout(0.2)
        # batch normalizations
        self.batchnorm2d1 = nn.BatchNorm2d(48)
        self.batchnorm2d2 = nn.BatchNorm2d(96)
        self.batchnorm2d3 = nn.BatchNorm2d(192)
        self.batchnorm2d4 = nn.BatchNorm2d(256)
        self.batchnorm1d1 = nn.BatchNorm1d(1500)
        self.batchnorm1d2 = nn.BatchNorm1d(750)
        self.batchnorm1d3 = nn.BatchNorm1d(300)
        self.batchnorm1d4 = nn.BatchNorm1d(100)
        self.batchnorm1d5 = nn.BatchNorm1d(50)
        # fully connected layers
        self.fc1 = nn.Linear(256*8*8, 1500)
        self.fc2 = nn.Linear(1500, 750)
        self.fc3 = nn.Linear(750, 300)
        self.fc4 = nn.Linear(300, 100)
        self.fc5 = nn.Linear(100, 50)
        self.fc6 = nn.Linear(50, 10)

    def forward(self, x):
        x = x  # input layer

        x = self.batchnorm2d1(self.conv1(x))
        x = F.relu(x)
        x = self.batchnorm2d2(self.conv2(x))
        x = F.relu(x)
        x = self.pool2(x)
        x = self.drop(x)
        x = self.batchnorm2d3(self.conv3(x))
        x = F.relu(x)
        x = self.batchnorm2d4(self.conv4(x))
        x = F.relu(x)
        x = self.pool2(x)
        x = self.drop(x)
        x = x.view(-1, 256*8*8)
        x = self.batchnorm1d1(self.fc1(x))
        x = F.relu(x)
        x = self.batchnorm1d2(self.fc2(x))
        x = F.relu(x)
        x = self.drop(x)
        x = self.batchnorm1d3(self.fc3(x))
        x = F.relu(x)
        x = self.batchnorm1d4(self.fc4(x))
        x = F.relu(x)
        x = self.drop(x)
        x = self.batchnorm1d5(self.fc5(x))
        x = F.relu(self.fc6(x))
        # torch.nnCrossEntropyLoss() contain nn.LogSoftmax()
        # so there is no need to use softmax
        return x

    def fit(self, train_loader, optimizer, epochs, device='cpu', valid_loader=None, verbose=5):
        '''
            verbose is number of batch report prints for epoch. For example
            verbose=0 will not print any batch reports, whereas verbose=5 will
            print 5 batch reports per epoch.
        '''

        assert epochs > 0, "epochs cannot be smaller then 1"
        self.to(device)
        criterion = nn.CrossEntropyLoss()
        valid_accuracies = []

        for epoch in range(epochs):
            epoch_loss = []          # Array of losses in epoch
            correct_predictions = 0  # Number of correct predictions made in validation
            total_predictions = 0    # Number of total predictions made in validation

            for i, data in enumerate(train_loader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss.append(loss.item())

                # torch.linspace() is creating 1d tensor with even spaces in between, after removing first and last values
                # we have wanted amount of evenly spaced values between two boundaries
                if i in torch.linspace(0, len(train_loader), verbose+2, dtype=torch.int32)[1:-1] and verbose != 0:
                    # This is batch report print
                    print(f'Epoch {epoch+1:3}, Batch {i+1:5}, loss {loss.item():.5f}')

            if valid_loader is not None:
                self.eval()
                with torch.no_grad():
                    for data in valid_loader:
                        images, labels = data
                        images, labels = images.to(device), labels.to(device)

                        outputs = self(images)
                        _, predicted = torch.max(outputs, 1)
                        c = (predicted == labels).squeeze()
                        total_predictions += len(c)
                        correct_predictions += c.sum()
                self.train()

            valid_accuracy = (correct_predictions / total_predictions) * 100
            valid_accuracies.append(valid_accuracy)

            valid_result = f', Validation Accuracy {valid_accuracy:6.2f}' if valid_loader is not None else ''
            print(f'Epoch {epoch+1:3}, loss {sum(epoch_loss) / len(epoch_loss):.5f}{valid_result}\n')

        return max(valid_accuracies)

    def test(self, test_loader, device='cpu'):
        '''
            Returns to a dictionary that contains the both
            categorical and total accuracies
        '''
        self.eval()
        self.to(device)

        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))

        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)

                outputs = self(images)
                _, predicted = torch.max(outputs, 1)
                c = (predicted == labels).squeeze()
                for i in range(len(images)):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

        self.train()

        for i in range(10):
            print(f'Accuracy of {self.classes[i]:5}: {100 * (class_correct[i]/class_total[i]):6.2f}%')
        print(f'Total accuracy is: {100 * (sum(class_correct)/sum(class_total)):6.2f}%')

    def save(self, path='cifar_net.pth'):
        torch.save(self.state_dict(), path)
