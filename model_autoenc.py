#!/usr/bin/env python
# coding: utf-8

# In[3]:


# import packages
import torch
import torch.nn as nn

# make model class
class autoencoder(nn.Module):
  def __init__(self):
    super(autoencoder, self).__init__()
    # encoder layers
    self.encoder= nn.Sequential(
        # first conv block
        nn.Conv2d(in_channels= 1, out_channels= 64, kernel_size= 5, stride= 2, padding= 2),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size= 2),
        # second conv block
        nn.Conv2d(in_channels= 64, out_channels= 128, kernel_size= 3, stride= 1, padding= 1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size= 2),
        # third conv block
        nn.Conv2d(in_channels= 128, out_channels= 256, kernel_size= 3, stride= 1, padding= 1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size= 2)
    )
        # decoder layers
    self.decoder= nn.Sequential(
        # first transpose conv block
        nn.ConvTranspose2d(in_channels= 256, out_channels= 200, kernel_size= 2, stride= 2),
        nn.ReLU(),
        # second transpose conv block
        nn.ConvTranspose2d(in_channels= 200, out_channels= 150, kernel_size= 2, stride= 2),
        nn.ReLU(),
        # third conv block
        nn.ConvTranspose2d(in_channels= 150, out_channels= 100, kernel_size= 2, stride= 2),
        nn.ReLU(),
        # fourth conv block
        nn.ConvTranspose2d(in_channels= 100, out_channels= 1, kernel_size= 2, stride= 2),
        nn.Sigmoid()
    )
  def forward(self, x):
    x= self.encoder(x)
    x= self.decoder(x)
    return x


