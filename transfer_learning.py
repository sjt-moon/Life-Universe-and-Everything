###########################
#####CODE REFERENCE SOURCE
##1. http://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
##2.https://medium.com/@airconaaron/visualizing-cnn-filters-in-pytorch-491f38a99ee5
###########################
from torchvision import transforms #add this line in the above snippet
from torch.utils.data import DataLoader #add this line in the above snippet
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from caltech256 import Caltech256
from torchvision.utils import make_grid

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
caltech256_train = Caltech256("/datasets/Caltech256/256_ObjectCategories/", data_transforms['train'], train=True)
train_data = DataLoader(
    dataset = caltech256_train,
    batch_size = 32,
    shuffle = True,
    num_workers = 4
)

caltech256_test = Caltech256("/datasets/Caltech256/256_ObjectCategories/", data_transforms['test'], train=False)
test_data = DataLoader(
    dataset = caltech256_test,
    batch_size = 32,
    shuffle = True,
    num_workers = 4
)
use_gpu = torch.cuda.is_available()

## Train the model

def train_model(model, criterion, optimizer, scheduler, num_epochs=50):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_acc = []
    train_loss= []
    test_acc = []
    test_loss= []
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        #train mode
        scheduler.step()
        model.train(True)  # Set model to training mode
        running_loss = 0.0
        running_corrects = 0
        for i, data in enumerate(train_data, 0):
            # get the inputs
            inputs, labels = data
            labels = labels.long().squeeze(1) - 1
            #labels = labels.long()
            # wrap them in Variable
            if use_gpu:
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            
            # forward
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            
            loss = criterion(outputs, labels)
            
            # zero the parameter gradients
            optimizer.zero_grad()



            # backward + optimize only if in training phase
            loss.backward()
            optimizer.step()

            # statistics
            
            running_loss += loss.data[0] * inputs.size(0)
            #print(running_loss)
            #print (i, ":")
            #print ("pred:", preds)
            #print ("labels", labels.data)
            running_corrects += torch.sum(preds == labels.data)
            #print ("running_corrects:", running_corrects)
        
        epoch_loss = running_loss / len(caltech256_train)
        epoch_acc = running_corrects / len(caltech256_train)
        train_acc.append(epoch_acc)
        train_loss.append(epoch_loss)
        print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            'train', epoch_loss, epoch_acc))

        #test mode
        model.train(False)  
        running_loss = 0.0
        running_corrects = 0
        for i, data in enumerate(test_data, 0):
            # get the inputs
            inputs, labels = data
            labels = labels.long().squeeze(1) - 1
            #labels = labels.long()
            # wrap them in Variable
            if use_gpu:
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            # zero the parameter gradients
            #optimizer.zero_grad()

            # forward
            outputs = model(inputs)
            #print(inputs.size(), labels.size(), outputs.size())
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)

            # statistics
            running_loss += loss.data[0] * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(caltech256_test)
        epoch_acc = running_corrects / len(caltech256_test)
        test_acc.append(epoch_acc)
        test_loss.append(epoch_loss)
        print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            'test', epoch_loss, epoch_acc))

        # deep copy the model
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
    print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, train_loss, test_loss, train_acc, test_acc

##Training hte last layer
model_conv = torchvision.models.vgg16(pretrained=True)
for param in model_conv.parameters():
    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
model_conv.classifier._modules['6'] = nn.Linear(4096, 256)

if use_gpu:
    model_conv = model_conv.cuda()

criterion = nn.CrossEntropyLoss()

# Observe that only parameters of final layer are being optimized as
# opoosed to before.
optimizer_conv = optim.SGD(model_conv.classifier._modules['6'].parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=10, gamma=0.1)

#Train and Evaluate
model_conv, train_loss, test_loss, train_acc, test_acc = train_model(model_conv, criterion, optimizer_conv,
                         exp_lr_scheduler, num_epochs=25)
##accuracy and loss
plt.figure(dpi=200)
plt.plot(train_loss,label = "train")
plt.plot(test_loss,label = "test")
plt.legend()
plt.show()
plt.figure(dpi=200)
plt.plot(train_acc, label = "train")
plt.plot(test_acc, label = "test")
plt.legend()
plt.show()

##filter visualization
model_conv = torchvision.models.vgg16(pretrained=True)
kernel = model_conv.features[0].weight.data.clone().numpy()
plt.figure(figsize=(10,10))
for idx, filt  in enumerate(kernel):
    plt.subplot(8,8, idx + 1)
    plt.imshow(filt[0, :, :])
    plt.axis('off')
     
plt.show()

##activation visualization
class Modified_VGG_plot(nn.Module):
    def __init__(self):
        super(Modified_VGG_plot, self).__init__()
        self.features = nn.Sequential(
            # stop at block4
            *list(model_conv.features.children())[:2]
        )

    def forward(self, x):
        x = self.features(x)
        return x

data, label = next(iter(train_data))
plt.figure(figsize=(10,10))
plt.imshow(data[0][0].numpy(),cmap="gray")
plt.show()


##Feature Extraction
##3 blocks
model_conv = torchvision.models.vgg16(pretrained=True)
class Modified_VGG3(nn.Module):
            def __init__(self):
                super(Modified_VGG3, self).__init__()
                self.features = nn.Sequential(
                    # stop at block3
                    *list(model_conv.features.children())[:17]
                )
                self.classifier = nn.Sequential(nn.Linear(200704, 1024), nn.Linear(1024,256))

            def forward(self, x):
                x = self.features(x)
                x = x.view(x.size(0), -1)
                x = self.classifier(x)
                return x

model_conv_3 = Modified_VGG3()
#ConvNet as fixed feature extractor
for param in model_conv_3.features.parameters():
    param.requires_grad = False
for param in model_conv_3.classifier.parameters():
    param.requires_grad = True
    
if use_gpu:
    model_conv_3 = model_conv_3.cuda()

criterion = nn.CrossEntropyLoss()

#optimizer_conv = optim.SGD(model_conv_3.classifier.parameters(), lr=0.00001, momentum=0.9)
optimizer_conv = optim.Adam(model_conv_3.classifier.parameters(), lr=0.00001, betas=(0.9, 0.999), weight_decay=0.1)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=10, gamma=0.1)

#Train and Evaluate
model_conv_3, train_loss_3, test_loss_3, train_acc_3, test_acc_3= train_model(model_conv_3, criterion, optimizer_conv,
                         exp_lr_scheduler, num_epochs=40)

plt.figure(dpi=200)
plt.plot(train_loss_3,label = "train")
plt.plot(test_loss_3,label = "test")
plt.legend()
plt.show()
plt.figure(dpi=200)
plt.plot(train_acc_3, label = "train")
plt.plot(test_acc_3, label = "test")
plt.legend()
plt.show()

##four blocks
class Modified_VGG4(nn.Module):
            def __init__(self):
                super(Modified_VGG4, self).__init__()
                self.features = nn.Sequential(
                    # stop at block4
                    *list(model_conv.features.children())[:24]
                )
                self.classifier = nn.Sequential(nn.Linear(100352, 1024), nn.Linear(1024 ,256))

            def forward(self, x):
                x = self.features(x)
                x = x.view(x.size(0), -1)
                x = self.classifier(x)
                return x

model_conv_4 = Modified_VGG4()
for param in model_conv_4.features.parameters():
    param.requires_grad = False
for param in model_conv_4.classifier.parameters():
    param.requires_grad = True

if use_gpu:
    model_conv_4 = model_conv_4.cuda()

criterion = nn.CrossEntropyLoss()

optimizer_conv = optim.Adam(model_conv_4.classifier.parameters(), lr=0.00001, betas=(0.9, 0.999), weight_decay=0.1)
# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=10, gamma=0.1)

#Train and Evaluate
model_conv_4, train_loss_4, test_loss_4, train_acc_4, test_acc_4 = train_model(model_conv_4, criterion, optimizer_conv,
                         exp_lr_scheduler, num_epochs=50)

plt.figure(dpi=200)
plt.plot(train_loss_4,label = "train")
plt.plot(test_loss_4,label = "test")
plt.legend()
plt.show()
plt.figure(dpi=200)
plt.plot(train_acc_4, label = "train")
plt.plot(test_acc_4, label = "test")
plt.legend()
plt.show()
























