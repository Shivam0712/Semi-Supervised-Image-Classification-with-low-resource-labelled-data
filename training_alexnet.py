#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().run_line_magic('matplotlib', 'inline')
#get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

import matplotlib.pyplot as plt

import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import datetime
import os
import sys
from IPython.display import display, clear_output
from torch.utils.data.sampler import SubsetRandomSampler
import shutil


# In[2]:

print(datetime.datetime.now(), ": Start")
data_dir = '/scratch/skp454/SPDL/data/ssl_data_96/supervised/train'
traindir = '/scratch/skp454/SPDL/data/ssl_data_96/supervised/train'
testdir = '/scratch/skp454/SPDL/data/ssl_data_96/supervised/val'
print(datetime.datetime.now(), ": Set Directories")

# In[ ]:


# def load_split_train_test(datadir, valid_size = .2):
#     train_transforms = transforms.ToTensor()
                                    

#     test_transforms = transforms.ToTensor()

#     train_data = datasets.ImageFolder(datadir, transform=train_transforms)
#     test_data = datasets.ImageFolder(datadir, transform=test_transforms)

#     num_train = len(train_data)
#     indices = list(range(num_train))
#     split = int(np.floor(valid_size * num_train))
#     np.random.shuffle(indices)
#     from torch.utils.data.sampler import SubsetRandomSampler
#     train_idx, test_idx = indices[split:], indices[:split]
#     train_sampler = SubsetRandomSampler(train_idx)
#     test_sampler = SubsetRandomSampler(test_idx)
#     trainloader = torch.utils.data.DataLoader(train_data, sampler=train_sampler, batch_size=64)
#     testloader = torch.utils.data.DataLoader(test_data, sampler=test_sampler, batch_size=64)
#     return trainloader, testloader

# trainloader, testloader = load_split_train_test(data_dir, .2)
# print(trainloader.dataset.classes)


# In[3]:


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


# In[4]:

def print_output(out_path, output_list):
    """
    Function that creates an output file with all of the output lines in
    the output list.
    Args:
        out_path: path to write the output file.
        output_list: list containing all the output items 
    """
    output_file = open(out_path,"a+")
    output_file.write("\n\n")
    output_file.write("\n".join(output_list))
    output_file.close()


def load_split_train_test(traindir, testdir):
    train_transforms = transforms.Compose([transforms.Resize(256),
                                       transforms.ToTensor(),
                                       ])
    test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.ToTensor(),
                                      ])

    train_data = datasets.ImageFolder(traindir,       
                    transform=train_transforms)
    test_data = datasets.ImageFolder(testdir,
                    transform=test_transforms)
    
    train_indices = list(range(len(train_data)))
    test_indices = list(range(len(train_data)))
    
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)
    
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(train_indices)
    
    trainloader = torch.utils.data.DataLoader(train_data, sampler=train_sampler, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_data, sampler=test_sampler, batch_size=1000)

    return trainloader, testloader

print(datetime.datetime.now(), ": loading data")
trainloader, testloader = load_split_train_test(traindir,testdir)
print(len(trainloader.dataset.classes))
print(datetime.datetime.now(), ": Data loaded")

# In[5]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# In[6]:


#import model
#model = model.Model()
print(datetime.datetime.now(), ": Loading Model")
model = models.alexnet(pretrained=True)
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

#criterion = nn.CrossEntropyLoss()
#optimizer = torch.optim.SGD(model.parameters(), lr = 0.001)
                                #,momentum=args.momentum , weight_decay=args.weight_decay)
model.to(device)

print(datetime.datetime.now(), ": Loading Model")
# In[7]:

print(datetime.datetime.now(), ": Begin Training")
epochs = 10
steps = 0
running_loss = 0
print_every = 1000
train_losses, test_losses = [], []
torch.manual_seed(100)
overall_best_accuracy = 0
for epoch in range(epochs):
    t0 = datetime.datetime.now()
    a = 0
    for inputs, labels in trainloader:
        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        t1 = datetime.datetime.now()
        a = a + 1
        #clear_output()
        print(str(t1), str(t1-t0), str(a/10), ' %; loss: ', str(loss.item())) 
    
    test_loss = 0
    accuracy = 0
    model.eval()
    with torch.no_grad():
        b = 0
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)
            test_loss += batch_loss.item()

            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            b = b + 1
            t2 = datetime.datetime.now()
            #clear_output()
            #print(str(t1), str(t1-t0), str(a/10), ' %; loss: ', str(loss))
            print(str(t2), str(t2-t1), str(b * 100/64), ' %; accuracy: ', str(accuracy/b))
            # remember best prec@1 and save checkpoint
            
    model.train()
        
    # remember best prec@1 and save checkpoint
    is_best = accuracy/len(testloader) > overall_best_accuracy
    overall_best_accuracy = max( overall_best_accuracy, accuracy/len(testloader))
    
    save_checkpoint({
        'epoch': epoch + 1,
        #'arch': args.arch,
        'state_dict': model.state_dict(),
        'best_acc': overall_best_accuracy,
        'optimizer' : optimizer.state_dict(),
    }, is_best)


    train_losses.append(running_loss/len(trainloader))
    test_losses.append(test_loss/len(testloader))                    
    lis =[]
    lis.append(str(datetime.datetime.now()))
    lis.append(f"Epoch {epoch+1}/{epochs}.. "
          f"Train loss: {running_loss/len(trainloader):.3f}.. "
          f"Test loss: {test_loss/len(testloader):.3f}.. "
          f"Test accuracy: {accuracy/len(testloader):.3f}")
    print(lis)
    print_output('logs.txt', lis)
    running_loss = 0
print(datetime.datetime.now(), ": End Training")

# In[ ]:

print(datetime.datetime.now(), ": Make Plots")
plt.plot(train_losses, label='Training loss')
plt.plot(test_losses, label='Validation loss')
plt.legend(frameon=False)
plt.savefig('Training_Performance')

print(datetime.datetime.now(), ": END")

