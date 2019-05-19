#!/usr/bin/env python
# coding: utf-8

# In[1]:


#from __future__ import print_function
import argparse
import os
import imp
import algorithms as alg
from dataloader import DataLoader, GenericDataset


# In[23]:


import torch
import torch.nn as nn
torch.cuda.device_count()


# In[3]:


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


# In[9]:


args_opt = Namespace(
    exp = 'Custom_RotNet_AlexNet',
    evaluate = False,
    checkpoint = 9,
    num_workers = 1,
    cuda = True,
    disp_step= 5
)


# In[10]:


exp_config_file = os.path.join('.','config',args_opt.exp+'.py')
# if args_opt.semi == -1:
exp_directory = os.path.join('.','experiments',args_opt.exp)
# else:
#    assert(args_opt.semi>0)
#    exp_directory = os.path.join('.','experiments/unsupervised',args_opt.exp+'_semi'+str(args_opt.semi))


# In[11]:


exp_config_file


# In[12]:


# Load the configuration params of the experiment
print('Launching experiment: %s' % exp_config_file)
config = imp.load_source("",exp_config_file).config
config['exp_dir'] = exp_directory # the place where logs, models, and other stuff will be stored
print("Loading experiment %s from file: %s" % (args_opt.exp, exp_config_file))
print("Generated logs, snapshots, and model files will be stored on %s" % (config['exp_dir']))


# In[13]:


# Set train and test datasets and the corresponding data loaders
data_train_opt = config['data_train_opt']
data_test_opt = config['data_test_opt']
num_imgs_per_cat = data_train_opt['num_imgs_per_cat'] if ('num_imgs_per_cat' in data_train_opt) else None


# In[14]:


dataset_train = GenericDataset(
    dataset_name=data_train_opt['dataset_name'],
    split=data_train_opt['split'],
    random_sized_crop=data_train_opt['random_sized_crop'],
num_imgs_per_cat=num_imgs_per_cat)


# In[15]:


dataset_test = GenericDataset(
    dataset_name=data_test_opt['dataset_name'],
    split=data_test_opt['split'],
    random_sized_crop=data_test_opt['random_sized_crop'])


# In[16]:


dloader_train = DataLoader(
    dataset=dataset_train,
    batch_size=data_train_opt['batch_size'],
    unsupervised=data_train_opt['unsupervised'],
    epoch_size=data_train_opt['epoch_size'],
    num_workers=args_opt.num_workers,
shuffle=True)


# In[17]:


dloader_test = DataLoader(
    dataset=dataset_test,
    batch_size=data_test_opt['batch_size'],
    unsupervised=data_test_opt['unsupervised'],
    epoch_size=data_test_opt['epoch_size'],
    num_workers=args_opt.num_workers,
shuffle=False)


# In[18]:


len(dloader_train.dataset)


# In[19]:


config['disp_step'] = args_opt.disp_step


# In[29]:


algorithm = getattr(alg, config['algorithm_type'])(config)


# In[30]:


# device = torch.device("cuda")
# device


# In[31]:


if args_opt.cuda: # enable cuda
#     if torch.cuda.device_count() > 1:
#         algorithm = nn.DataParallel(algorithm)
    algorithm.load_to_gpu()
#     algorithm.to(device)
if args_opt.checkpoint > 0: # load checkpoint
    algorithm.load_checkpoint(args_opt.checkpoint, train= (not args_opt.evaluate))


# In[ ]:


if not args_opt.evaluate: # train the algorithm
    algorithm.solve(dloader_train, dloader_test)
else:
    algorithm.evaluate(dloader_test) # evaluate the algorithm


# /scratch/um367/urwa-env/py2.7.12/lib/python2.7/site-packages/torchvision/transforms/functional.py", line 48, in to_tensor
