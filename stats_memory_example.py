import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import math
import sys
import numpy as np
import pdb
import os 
import matplotlib.pyplot as plt
import time
import pandas as pd

device = torch.device("cuda")

def saveVals(self, grad_input, grad_output):
    with torch.no_grad():
        if(self.convLayer):
            batchSize, neurons, xDim, yDim = grad_output[0].size()
        else:
            batchSize, neurons = grad_output[0].size()
            xDim = 1
            yDim = 1
        self.current = grad_output[0].detach().double()
        self.statsLayer.values[0].current = (self.current - (self.averageTensor))
        self.statsLayer.statsNonlinearOuts[0].register_hook(lambda grad: self.statsLayer.values[0].current)
        if(self.convLayer):
            self.average = self.average * 0.99 + self.current.sum((0,2,3)) * 0.01
        else:
            self.average = self.average * 0.99 + self.current.sum(0) * 0.01
    

class newMod1(nn.Module):
    
    def __init__(self, convLayer, in_channels, out_channels, kernel_size = -1, 
                 stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
    
        super(newMod1, self).__init__()


        self.convLayer = convLayer

        #base options
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.padding_mode = padding_mode

        self.register_buffer('average', torch.Tensor(out_channels).zero_().to(device).double())
        
        if(convLayer):
            self.layer = nn.Conv2d(self.in_channels,
                                    self.out_channels,
                                    self.kernel_size,
                                    self.stride,
                                    self.padding,
                                    self.dilation,
                                    self.groups,
                                    self.bias,
                                    self.padding_mode).double()
        else:
            self.layer = nn.Linear(self.in_channels, self.out_channels, bias=self.bias).double()
        self.statsLayer = newMod2(self.convLayer, 
                                    self.in_channels,
                                    self.out_channels,
                                    self.kernel_size,
                                    self.stride,
                                    self.padding,
                                    self.dilation,
                                    self.groups,
                                    self.bias,
                                    self.padding_mode)
        
        self.register_backward_hook(saveVals)
    def forward(self, x):
        out = self.layer(x)
        statsCalculated = self.statsLayer(x)[0]
        self.averageTensor = self.statsLayer.values[0].average
        #this just adds it to the graphs so things get calculated
        out = out + statsCalculated
        return out
    

        
class valueTracker():
    def __init__(self, out_channels, convLayer):
        if(torch.cuda.is_available()):
            self.current = torch.Tensor(1).to(device).double()
            self.mod2Outs = torch.Tensor(1).to(device).double()
            self.average = torch.Tensor(out_channels).zero_().to(device).double()
            self.averageMult = torch.Tensor(out_channels).zero_().to(device).double()
            self.convLayer = convLayer

def Tagger(inp, Values):
    class Tagger(torch.autograd.Function):
        @staticmethod
        def forward(ctx, inp):
            return inp
        @staticmethod
        def backward(ctx, grad_out):
            with torch.no_grad():
                savedValues = Values
                multiplied = savedValues.mod2Outs * (savedValues.current)
                if(savedValues.convLayer):
                    multiplied = multiplied.sum(((0,2,3)))
                else:
                    multiplied = multiplied.sum(0)
                grad_in = grad_out.detach()
                return grad_in, None
    return Tagger.apply(inp)


class noBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp):
        return inp
    @staticmethod
    def backward(ctx, grad_out):
        return grad_out * 0     


class noForward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp):
        return inp * 0
    @staticmethod
    def backward(ctx, grad_out):
        return grad_out     
    
    

class newMod2(nn.Module):
    def __init__(self, convLayer, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super(newMod2, self).__init__()
        
        self.convLayer = convLayer
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.padding_mode = padding_mode
        

        self.layers = nn.ModuleList([])
        self.values = {}
        with torch.no_grad():
            if(self.convLayer):
                self.layers.append(nn.Conv2d(self.in_channels,
                                                        self.out_channels,
                                                        self.kernel_size,
                                                        self.stride,
                                                        self.padding,
                                                        self.dilation,
                                                        self.groups,
                                                        self.bias,
                                                        self.padding_mode).double())
            else:
                self.layers.append(nn.Linear(self.in_channels, self.out_channels, bias=self.bias).double())
        self.layers[0].to(device)
        self.values[0] = valueTracker(out_channels, self.convLayer)
        if(self.convLayer):
            self.values[0].average = torch.tensor(out_channels).double().view(1,-1,1,1).detach().clone().to(device)
        else:
            self.values[0].average = torch.tensor(out_channels).double().view(1,-1).detach().clone().to(device)
        self.values[0].average.requires_grad = False

        
    def forward(self, x):
        outs = {}
        noback = noBackward.apply
        x = noback(x)
        statsOuts = {}
        self.statsNonlinearOuts = {}
        statsOuts[0] = self.layers[0](x)
        statsOuts[0] = Tagger(statsOuts[0], self.values[0])
        self.statsNonlinearOuts[0] = F.relu(statsOuts[0])
        self.values[0].mod2Outs = self.statsNonlinearOuts[0]
        noforward = noForward.apply
        statsOuts[0] = noforward(self.statsNonlinearOuts[0])
                
        return statsOuts
    
    
