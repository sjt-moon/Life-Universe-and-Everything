# Reference
# https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb


import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

class Trainer():
    def __init__(self, model, criterion='cross-entropy', optimizer='sgd'):
        assert model.parameters()!=None
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.opt_name = optimizer
        #self.optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        #self.max_iter = max_iter

        self.best_model_para = None
        self.best_validation_loss = 99999

    def fit(self, train, valid, test, use_gpu=True, early_stopping=-1, weight_adj=False, lr=0.1, max_iter=100):
        assert isinstance(train, torch.utils.data.dataloader.DataLoader)
        assert isinstance(valid, torch.utils.data.dataloader.DataLoader)
        assert isinstance(test, torch.utils.data.dataloader.DataLoader)
        if use_gpu:
            self.model.cuda()

        if self.opt_name=='sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        elif self.opt_name=='adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        else:
            print('Invalid optimizer')
            return

        self.max_iter = max_iter
        training_loss, validation_loss, test_loss = [], [], []
        training_accu, validation_accu, test_accu = [], [], []

        for epoch in range(1, self.max_iter+1):
            print('================\nEpoch: %d\n' % (epoch,))
            running_loss = 0.0
            for i, xi in enumerate(train, 1):
                # get input
                inputs, labels = xi
                if use_gpu:
                    inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

                # zero grads
                self.optimizer.zero_grad()

                # forward
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                # backward
                loss.backward()
                self.optimizer.step()
                running_loss += loss.data[0]

                # log
                #training_loss.append(loss.data[0])
                if i%100==0:
                    print('Epoch %d\tBatch %d loss: %.3f' % (epoch, i, running_loss/100.0))
                    running_loss = 0.0

            training_loss.append(loss.data[0])
            train_accu, _, __ = self.predict(train, use_gpu) 
            training_accu.append(train_accu)

            # validation
            valid_accu, _, validation_loss_i = self.predict(valid, use_gpu)
            validation_loss.append(validation_loss_i)
            validation_accu.append(valid_accu)

            # test
            test_accu_i, _, test_loss_i = self.predict(test, use_gpu)
            test_loss.append(test_loss_i)
            test_accu.append(test_accu_i)

            
            # early stopping
            if early_stopping>0:
                #_, __, validation_loss_i = self.predict(valid, use_gpu)
                #validation_loss.append(validation_loss_i)
                if validation_loss_i < self.best_validation_loss:
                    self.best_model_para = self.model.state_dict()
                    self.best_validation_loss = validation_loss_i
                if is_increasing_consistently(validation_loss[-early_stopping:]):
                    print('Early stop at %d epoch' % epoch)
                    break
            

            # weight adjusting
            # scale weights to N(1,1), the lower accuracy for a class, the higher weight it gets
            if weight_adj:
                _, class_accuracy, __ = self.predict(train, use_gpu)
                weights = np.array(class_accuracy)

                # suppose each class has an accuracy of p, there are 10 classes
                # weight = (1-p/(10*p)) * 10/9 = 1
                # thus we could compare loss with those without weight adjusting
                weights = (1 - weights / np.sum(weights)) * 10.0 / 9
                weights = torch.FloatTensor(weights)
                if use_gpu:
                    weights = weights.cuda()
                self.criterion = nn.CrossEntropyLoss(weight=weights)

        # recover the best model on validation set
        if early_stopping>0:
            self.model.load_state_dict(self.best_model_para)

        return training_loss, training_accu, validation_loss, validation_accu, test_loss, test_accu

    def predict(self, inputs, use_gpu=True):
        '''Return overall accuracy & accuracy for each class & loss.'''
        assert isinstance(inputs, torch.utils.data.dataloader.DataLoader)

        correct, total = 0, 0
        class_correct, class_total = [0 for i in range(10)], [0 for j in range(10)]
        loss = nn.CrossEntropyLoss()
        running_loss = 0.0
        
        for xi in inputs:
            images, labels = xi
            if use_gpu:
                images = images.cuda()
                labels = labels.cuda()
            images = Variable(images, requires_grad=False)        
            labels_var = Variable(labels, requires_grad=False)

            outputs = self.model(images)
            running_loss += loss(outputs, labels_var).data[0]

            _, y_preds = torch.max(outputs.data, 1)
            c = (y_preds == labels).squeeze()

            # accuracy for each class
            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += c[i]
                class_total[label] += 1

            # overall accuracy
            correct += c.sum()
            total += labels.size(0)

        accuracy = 100 * correct / total
        class_accuracy = [100 * class_correct[i] / class_total[i] for i in range(10)]
        running_loss /= 100.0       # 100 mini-batches
        return accuracy, class_accuracy, running_loss 

def load(batch_size=100, scale=32):
    transform = transforms.Compose([
                transforms.Scale(scale),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
    validset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
    indices = list(range(len(trainset)))
    np.random.shuffle(indices)
    train_idx, valid_idx = indices[:45000], indices[45000:]

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                        shuffle=False, num_workers=2,
                                        sampler=SubsetRandomSampler(train_idx))
    
    validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size,
                                        shuffle=False, num_workers=2,
                                        sampler=SubsetRandomSampler(valid_idx))

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, validloader, testloader, classes

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def is_increasing_consistently(arr):
    if len(arr)<3:
        return False
    for i in range(len(arr)-1):
        if arr[i+1]<arr[i]:
            return False
    return True

