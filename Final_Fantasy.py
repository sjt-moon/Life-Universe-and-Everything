
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


# In[8]:


importlib.reload(utils)


# In[2]:


train, valid, test, classes = utils.load(scale=32)


# # LenNet

# In[9]:


lennet = Life_Universe_and_Everything.LenNet()
coach_len = utils.Trainer(lennet)
len_tr_loss1, len_tr_ac1, len_te_loss1, len_te_ac1 = coach_len.fit(train=train, valid=test, early_stopping=3, lr=0.1, max_iter=100)


# In[11]:


len_tr_loss2, len_tr_ac2, len_te_loss2, len_te_ac2 = coach_len.fit(train=train, valid=test, early_stopping=3, lr=0.01)

len_tr_loss3, len_tr_ac3, len_te_loss3, len_te_ac3 = coach_len.fit(train=train, valid=test, early_stopping=3, lr=0.001)

torch.save(lennet.state_dict(), './final/LenNet-ep.pkl')


# In[13]:


lennet_tr_loss = len_tr_loss1 + len_tr_loss2 + len_tr_loss3
lennet_tr_accu = len_tr_ac1 + len_tr_ac2 + len_tr_ac3
lennet_te_loss = len_te_loss1 + len_te_loss2 + len_te_loss3
lennet_te_accu = len_te_ac1 + len_te_ac2 + len_te_ac3


# In[12]:


len_accu, len_class_accu, len_te_loss = coach_len.predict(test)
print('Accuracy @ test set: %.3f' % len_accu)

classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')
for c,a in zip(classes, len_class_accu):
    print(c+':\t%.3f%%' % a)


# In[14]:


t = [i for i in range(len(lennet_tr_loss))]
plt.plot(t, lennet_tr_loss, label='train')
plt.plot(t, lennet_te_loss, label='test')
plt.xlabel('# of epochs')
plt.ylabel('loss')
plt.title('LenNet Loss')
plt.legend()
plt.show()


# In[15]:


plt.plot(t, lennet_tr_accu, label='train')
plt.plot(t, lennet_te_accu, label='test')
plt.xlabel('# of epochs')
plt.ylabel('accuracy')
plt.title('LenNet Accuracy')
plt.legend()
plt.show()


# In[16]:


# record data -> file
with open('LenNet-results.log', 'w') as fw:
    titles = ['train-loss', 'test-loss', 'train-accu', 'test-accu']
    fw.write('\t'.join(titles)+'\n')
    for result in zip(lennet_tr_loss, lennet_te_loss, lennet_tr_accu, lennet_te_accu):
        result = [str(r) for r in result]
        fw.write('\t'.join(result)+'\n')


# In[3]:


def log(a,b,c,d,name):
    with open(name, 'w') as fw:
        titles = ['train-loss', 'test-loss', 'train-accu', 'test-accu']
        fw.write('\t'.join(titles)+'\n')
        for result in zip(a,b,c,d):
            result = [str(r) for r in result]
            fw.write('\t'.join(result)+'\n')


# # ResNet-18
# 
# 1-150: lr = 0.1
# 150-250: lr = 0.01
# 250-350: lr = 0.001

# In[4]:


resnet18 = Life_Universe_and_Everything.ResNet18()
resnet18.load_state_dict(torch.load('./final/ResNet18-2.pkl'))
coach_res = utils.Trainer(resnet18)
#res18_tr_loss1, res18_tr_ac1, res18_te_loss1, res18_te_ac1 = coach_res.fit(train=train, valid=test, lr=0.1)
#log(res18_tr_loss1, res18_tr_ac1, res18_te_loss1, res18_te_ac1, './final/resnet-1.log')
#torch.save(resnet18.state_dict(), './final/ResNet18-1.pkl')

#res18_tr_loss2, res18_tr_ac2, res18_te_loss2, res18_te_ac2 = coach_res.fit(train=train, valid=test, lr=0.01)
#log(res18_tr_loss2, res18_tr_ac2, res18_te_loss2, res18_te_ac2, './final/resnet-2.log')
#torch.save(resnet18.state_dict(), './final/ResNet18-2.pkl')

res18_tr_loss3, res18_tr_ac3, res18_te_loss3, res18_te_ac3 = coach_res.fit(train=train, valid=test, lr=0.001)
log(res18_tr_loss3, res18_tr_ac3, res18_te_loss3, res18_te_ac3, './final/resnet-3.log')

torch.save(resnet18.state_dict(), './final/ResNet18-3.pkl')


# In[ ]:


res_tr_loss = res18_tr_loss1 + res18_tr_loss2 + res18_tr_loss3
res_te_loss = res18_te_loss1 + res18_te_loss2 + res18_te_loss3
res18_tr_ac = res18_tr_ac1 + res18_tr_ac2 + res18_tr_ac3
res18_te_ac = res18_te_ac1 + res18_te_ac2 + res18_te_ac3


# In[ ]:


x = [i for i in range(len(res18_tr_ac))]
plt.plot(x, res18_tr_ac, label='train')
plt.plot(x, res18_te_ac, label='test')
plt.xlabel('# of epochs')
plt.ylabel('accuracy')
plt.title('ResNet-18 Accuracy')
plt.legend()
plt.show()


# In[ ]:


plt.plot(x, res_tr_loss, label='train')
plt.plot(x, res_te_loss, label='test')
plt.xlabel('# of epochs')
plt.ylabel('loss')
plt.title('ResNet-18 Loss')
plt.legend()
plt.show()


# In[3]:


resnet = Life_Universe_and_Everything.ResNet18()
#resnet.cuda()
resnet.load_state_dict(torch.load('ResNet18-32p-93.pkl'))


# In[5]:


resnet.cuda()
coach = utils.Trainer(resnet)

res_accu, res_class_accu, res_tee_loss = coach.predict(test)
print('Accuracy @ test set: %.3f' % res_accu)

classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')
for c,a in zip(classes, res_class_accu):
    print(c+':\t%.3f%%' % a)


# In[ ]:


# record data -> file
with open('ResNet-results.log', 'w') as fw:
    titles = ['train-loss', 'test-loss', 'train-accu', 'test-accu']
    fw.write('\t'.join(titles)+'\n')
    for result in zip(res_tr_loss, res_te_loss, res18_tr_ac, res18_te_ac):
        result = [str(r) for r in result]
        fw.write('\t'.join(result)+'\n')


# load data, draw for ResNet

# In[25]:


res_tr_loss, res_te_loss, res_tr_accu, res_te_accu = [], [], [], []

file = './final/resnet-'
for i in range(1,4):
    with open(file+str(i)+'.log') as fr:
        for line in fr.readlines():
            if line[0]=='t':
                continue
            line = line.strip().split('\t')
            line = [float(e) for e in line]
            res_tr_loss.append(line[0]) 
            res_tr_accu.append(line[1]) 
            res_te_loss.append(line[2]) 
            res_te_accu.append(line[3]) 
res_tr_accu = np.array(res_tr_accu)
res_te_accu = np.array(res_te_accu)
res_tr_loss = np.array(res_tr_loss)
res_te_loss = np.array(res_te_loss)


# In[26]:


x = [i for i in range(len(res_tr_loss))]
plt.plot(x, res_tr_loss, label='train')
plt.plot(x, res_te_loss, label='test')
plt.xlabel('# of epochs')
plt.ylabel('loss')
plt.title('ResNet-18 Loss')
plt.legend()
plt.show()


# In[27]:


plt.plot(x, res_tr_accu, label='train')
plt.plot(x, res_te_accu, label='test')
plt.xlabel('# of epochs')
plt.ylabel('accuracy')
plt.title('ResNet-18 Accuracy')
plt.legend()
plt.show()


# # Lenet
# 
# ## Xavier init with SGD

# In[10]:


importlib.reload(Life_Universe_and_Everything)
importlib.reload(utils)


# In[8]:


lenet = Life_Universe_and_Everything.LenNet()
coach = utils.Trainer(lenet)
x_tr_loss, x_tr_accu, x_va_loss, x_va_accu, x_te_loss, x_te_accu = coach.fit(train, valid, test, early_stopping=3, lr=0.01)


# ## Use both Adam optimizer and Xavier init

# In[9]:


lenet_a = Life_Universe_and_Everything.LenNet()
coach_a = utils.Trainer(lenet_a, optimizer='adam')
a_tr_loss, a_tr_accu, a_va_loss, a_va_accu, a_te_loss, a_te_accu = coach_a.fit(train, valid, test, early_stopping=3, lr=0.01)


# ## Non-init and SGD

# In[11]:


lenet_ns = Life_Universe_and_Everything.LenNet()
coach_ns = utils.Trainer(lenet_ns)
ns_tr_loss, ns_tr_accu, ns_va_loss, ns_va_accu, ns_te_loss, ns_te_accu = coach_ns.fit(train, valid, test, early_stopping=3, lr=0.01)


# ## Non-init and Adam

# In[12]:


lenet_na = Life_Universe_and_Everything.LenNet()
coach_na = utils.Trainer(lenet_na, optimizer='adam')
na_tr_loss, na_tr_accu, na_va_loss, na_va_accu, na_te_loss, na_te_accu = coach_na.fit(train, valid, test, early_stopping=3, lr=0.01)


# In[14]:


plt.plot(x_te_accu, label='Xavier SGD')
plt.plot(a_te_accu, label='Xavier Adam')
plt.plot(na_te_accu, label='Adam')
plt.plot(ns_te_accu, label='SGD')
plt.xlabel('# of iterations')
plt.ylabel('test accuracy')
plt.legend()
plt.show()

