
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import importlib
import torch
import torchvision

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import sys
sys.path.insert(0, './Life-Universe-and-Everything')

import utils
import Life_Universe_and_Everything


# In[5]:


importlib.reload(Life_Universe_and_Everything)
importlib.reload(utils)


# In[3]:


train, valid, test, classes = utils.load(scale=32)


# Before doing anything, let's have a look at label distributions.

# In[51]:


indices = {}
for i,c in enumerate(classes):
    indices[i] = (c, [])
    
cnt = 0
for batch_i in valid:
    x, label = batch_i
    for (xi,li) in zip(x, label):
        indices[li][1].append(cnt)
        cnt+=1


# In[52]:


for i,c in enumerate(classes):
    plt.plot(indices[i][1], label=c)
plt.legend()
plt.show()


# Test for LenNet

# In[58]:


lennet = Life_Universe_and_Everything.LenNet()
fitter1 = utils.Trainer(lennet, max_iter=300)
lennet_loss_over_tr, lennet_loss_val = fitter1.fit(train, valid)


# As training loss fluctuates fiercely, we average the near 5 mini-batches.

# In[31]:


#len_loss = [np.mean(lennet_loss_over_tr[5*i:5*i+5]) for i in range(int(len(lennet_loss_over_tr)/5))]


# let's view the last 20 validation loss

# In[54]:


#plt.plot([i for i in range(len(lennet_loss_over_tr))], lennet_loss_over_tr, label='train')
plt.plot([i for i in range(20)], lennet_loss_val[-20:], '-x')
#plt.plot([j for j in range(int(len(lennet_loss_val)/4))], [lennet_loss_val[4*k] for k in range(int(len(lennet_loss_val)/4))], '-x', label='valid')
plt.xlabel('# of mini-batches')
plt.ylabel('loss')
plt.title('LenNet training loss')
plt.legend()
plt.show()


# for training/validation loss over training (LenNet early stopping)

# In[62]:


plt.plot([i for i in range(len(lennet_loss_over_tr))], lennet_loss_over_tr, label='train')
#plt.plot([i for i in range(20)], lennet_loss_val[-20:], '-x')
plt.plot([j for j in range(len(lennet_loss_val))], lennet_loss_val, '-x', label='valid')
plt.xlabel('# of epochs')
plt.ylabel('loss')
plt.title('LenNet training loss')
plt.legend()
plt.show()


# In[59]:


accu, class_accu, te_loss = fitter1.predict(test)
print('Accuracy @ test set: %.3f' % accu)

classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')
for c,a in zip(classes, class_accu):
    print(c+':\t%.3f%%' % a)


# In[65]:


# log loss to file
with open('LenNet-loss-tr','w') as fw:
    for i in lennet_loss_over_tr:
        fw.write(str(i)+'\n')
with open('LenNet-loss-val','w') as fw:
    for i in lennet_loss_val:
        fw.write(str(i)+'\n')


# In[61]:


# Save the Trained Model
torch.save(lennet.state_dict(), 'LenNet.pkl')


# Test for ResNet-34
# 
# early stopping

# In[6]:


#import Life_Universe_and_Everything
importlib.reload(Life_Universe_and_Everything)
importlib.reload(utils)


# In[69]:


train, valid, test, classes = utils.load()


# In[70]:


resnet = Life_Universe_and_Everything.ResNet_34()
fitter = utils.Trainer(resnet, max_iter=300)


# In[71]:


resnet_loss_over_tr, resnet_loss_val = fitter.fit(train, valid)


# 224 x 224 ResNet-34 early stops at 11 epochs

# In[73]:


accu, class_accu, te_loss = fitter.predict(test)
print('Accuracy @ test set: %.3f' % accu)

classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')
for c,a in zip(classes, class_accu):
    print(c+':\t%.3f%%' % a)


# In[74]:


plt.plot([i for i in range(len(resnet_loss_over_tr))], resnet_loss_over_tr, label='train')
#plt.plot([i for i in range(20)], lennet_loss_val[-20:], '-x')
plt.plot([j for j in range(len(resnet_loss_val))], resnet_loss_val, '-x', label='valid')
plt.xlabel('# of epochs')
plt.ylabel('loss')
plt.title('LenNet training loss')
plt.legend()
plt.show()


# In[80]:


# Save the Trained Model
torch.save(resnet.state_dict(), 'ResNet-224-earlysp.pkl')


# 224 x 224 images, ResNet-34 300 iterations

# In[6]:


accu, class_accu = fitter.predict(test)
print('Accuracy @ test set: %.3f' % accu)


# In[7]:


classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')
for c,a in zip(classes, class_accu):
    print(c+':\t%.3f%%' % a)


# 32 x 32 images, ResNet-28 100 iterations

# In[42]:


accu, class_accu = fitter.predict(test)
print('Accuracy @ test set: %.3f' % accu)


# In[43]:


classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')
for c,a in zip(classes, class_accu):
    print(c+':\t%.3f%%' % a)


# In[8]:


# Save the Trained Model
torch.save(resnet.state_dict(), 'ResNet-34-224pixel-47iter.pkl')


# The following is ResNet-34 (224x224 input) with early stopping (11 epochs).

# In[78]:


plt.plot(lennet_loss_over_tr, label='LenNet')
plt.plot(resnet_loss_over_tr, label='ResNet-34')
plt.xlabel('# of epochs')
plt.ylabel('Training loss over training')
plt.legend()
plt.show()


# In[79]:


plt.plot(lennet_loss_val, label='LenNet')
plt.plot(resnet_loss_val, label='ResNet-34')
plt.xlabel('# of epochs')
plt.ylabel('Validation loss over training')
plt.legend()
plt.show()


# ### What if we adjust class weights dynamically?

# In[92]:


importlib.reload(Life_Universe_and_Everything)
importlib.reload(utils)


# In[94]:


train, valid, test, classes = utils.load(scale=32)


# In[95]:


lennet = Life_Universe_and_Everything.LenNet()
fitter1 = utils.Trainer(lennet, max_iter=300)
lennet_loss_over_tr, lennet_loss_val = fitter1.fit(train, valid, weight_adj=True)


# In[96]:


accu, class_accu, te_loss = fitter1.predict(test)
print('Accuracy @ test set: %.3f' % accu)

classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')
for c,a in zip(classes, class_accu):
    print(c+':\t%.3f%%' % a)


# In[98]:


lennet_loss_no_adj = []
with open('./LenNet-loss-tr') as fr:
    for line in fr.readlines():
        line = float(line.strip())
        lennet_loss_no_adj.append(line)


# We could see that LenNet-adjust does NOT perform better than naive algorithm, by plotting weights v.s. class accuracy, I believe it's the result of our weight equation. weight = (1 - accuracy / sum(accuracy)) * 10 / 9 does not give classes with low accuracy (e.g. cat) too much attention.

# In[100]:


plt.plot(lennet_loss_no_adj, label='LenNet-naive')
plt.plot(lennet_loss_over_tr, label='LenNet-adjust')
plt.xlabel('# of epochs')
plt.ylabel('Training loss over training')
plt.legend()
plt.show()


# In[104]:


for c,a,w in zip(classes, class_accu, fitter1.criterion.weight):
    print(c+':\t%.3f%%' % a+'\t '+str(w))


# what if we use ResNet-adjust?

# In[18]:


importlib.reload(Life_Universe_and_Everything)
importlib.reload(utils)


# In[50]:


train, valid, test, classes = utils.load()


# In[20]:


resnet = Life_Universe_and_Everything.ResNet_34()
fitter = utils.Trainer(resnet, max_iter=300)
resnet_loss_over_tr, resnet_loss_val = fitter.fit(train, valid, weight_adj=True)


# In[21]:


accu, class_accu, te_loss = fitter.predict(test)
print('Accuracy @ test set: %.3f' % accu)

classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')
for c,a in zip(classes, class_accu):
    print(c+':\t%.3f%%' % a)


# In[24]:


resnet_loss_val


# In[51]:


_,__,val = fitter.predict(valid)
print(val)


# In[25]:


fitter.best_validation_loss


# In[53]:


_,__,val = fitter.predict(valid)
print(val)


# In[54]:


tt = []
for i in valid:
    tt.append(i)


# In[52]:


t = []
for i in valid:
    t.append(i)


# In[55]:


t[0][0]


# In[56]:


tt[0][0] 


# test area

# In[58]:


len(t) == len(tt)


# ### TEST

# In[2]:


import torchvision.transforms as transforms


# In[3]:


transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# 0 - 150 epochs, lr = 0.1

# In[4]:


resnet18 = Life_Universe_and_Everything.ResNet18()


# In[5]:


coach = utils.Trainer(resnet18, max_iter=150)
loss_1, _ = coach.fit(train=trainloader, early_stopping=999999)


# In[6]:


# Save the Trained Model
torch.save(resnet18.state_dict(), 'ResNet-18-32pixel-1.pkl')


# In[7]:


coach2 = utils.Trainer(resnet18, max_iter=100)
loss_2, _ = coach2.fit(train=trainloader, early_stopping=999999, lr=0.01)


# In[8]:


# Save the Trained Model
torch.save(resnet18.state_dict(), 'ResNet-18-32pixel-2.pkl')


# In[9]:


coach3 = utils.Trainer(resnet18, max_iter=100)
loss_3, _ = coach3.fit(train=trainloader, early_stopping=999999, lr=0.001)


# In[10]:


# Save the Trained Model
torch.save(resnet18.state_dict(), 'ResNet-18-32pixel-3.pkl')


# In[29]:


loss_all = loss_1 + loss_2 + loss_3


# In[30]:


plt.plot([i for i in range(len(loss_all))], loss_all)
plt.show()


# In[11]:


accu, class_accu, te_loss = coach3.predict(testloader)
print('Accuracy @ test set: %.3f' % accu)

classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')
for c,a in zip(classes, class_accu):
    print(c+':\t%.3f%%' % a)


# In[22]:


tmp_model = Life_Universe_and_Everything.ResNet18()
tmp_model.load_state_dict(torch.load('ResNet-18-32pixel-3.pkl'))


# In[26]:


tmp_model.cuda()


# In[27]:


tmp_coach = utils.Trainer(tmp_model)
accut, class_accut, te_losst = tmp_coach.predict(testloader)


# In[28]:


accut

