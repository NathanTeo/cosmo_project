"""
Author: Nathan Teo

This script contains generators and discriminators (or critics) used in the GANs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import functools
from torch.nn.utils import spectral_norm
import math
import numpy as np

import code_model.layers as layers



"""Discriminators"""
class Discriminator_v1(nn.Module):
    """
    Simple discriminator based on standard convolutional neural network architechture
    """
    def __init__(self, **training_params):
        super().__init__()
        # Params
        network_params = training_params['network_params']
        
        image_channels = network_params['image_channels']
        conv_size = network_params['discriminator_conv_size']
        conv_dropout = network_params['conv_dropout']
        linear_size = network_params['discriminator_linear_size']
        linear_dropout = network_params['linear_dropout']
        
        image_size = training_params['image_size']
        self.model_version = training_params['model_version']
        
        # Simple CNN
        self.cnn = nn.Sequential(
            nn.Conv2d(image_channels, conv_size, kernel_size=5, stride=1), 
            nn.BatchNorm2d(conv_size), nn.ReLU(inplace=True), nn.Dropout2d(conv_dropout),

            nn.Conv2d(conv_size, conv_size*2, kernel_size=5, stride=1), 
            nn.BatchNorm2d(conv_size*2), nn.ReLU(inplace=True), nn.Dropout2d(conv_dropout),
            
            nn.Flatten()
        )
        
        # Classifier
        n_channels = self.cnn(torch.empty(1, 1, image_size, image_size)).size(-1)
        
        self.classifier = nn.Sequential(
            nn.Dropout(linear_dropout),
            nn.Linear(n_channels, linear_size),
            nn.ReLU(),
            nn.Dropout(linear_dropout),
            nn.Linear(linear_size, 1)
    )
  
    def forward(self, x):
        x = self.cnn(x)
        x = self.classifier(x)
        
        if self.model_version=='CGAN':
            x = torch.sigmoid(x)
            
        return x

class Discriminator_v2(nn.Module):
    """
    Simple discriminator that uses instance norm and leaky ReLU
    """
    def __init__(self, **training_params):
        super().__init__()
        # Params
        network_params = training_params['network_params']
        
        image_channels = network_params['image_channels']
        conv_size = network_params['discriminator_conv_size']
        conv_dropout = network_params['conv_dropout']
        linear_size = network_params['discriminator_linear_size']
        linear_dropout = network_params['linear_dropout']

        image_size = training_params['image_size']
        
        self.model_version = training_params['model_version']
        
        # Simple CNN
        self.cnn = nn.Sequential(
            nn.Conv2d(image_channels, conv_size, kernel_size=5, stride=1), 
            nn.InstanceNorm2d(conv_size, affine=True), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(conv_dropout),

            nn.Conv2d(conv_size, conv_size*2, kernel_size=5, stride=1),
            nn.InstanceNorm2d(conv_size*2, affine=True), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(conv_dropout),
            
            nn.Flatten()
        )
        
        # Classifier
        n_channels = self.cnn(torch.empty(1, 1, image_size, image_size)).size(-1)
        
        self.classifier = nn.Sequential(
            nn.Dropout(linear_dropout),
            nn.Linear(n_channels, linear_size),
            nn.ReLU(),
            nn.Dropout(linear_dropout),
            nn.Linear(linear_size, 1)
    )
  
    def forward(self, x):
        x = self.cnn(x)
        x = self.classifier(x)
        
        if self.model_version=='CGAN':
            x = torch.sigmoid(x)
        
        return x
    
class Discriminator_v3(nn.Module):
    """
    Discriminator that uses instance norm and leaky ReLU
    """
    def __init__(self, **training_params):
        super().__init__()
        # Params
        network_params = training_params['network_params']
        
        image_channels = network_params['image_channels']
        conv_size = network_params['discriminator_conv_size']
        conv_dropout = network_params['conv_dropout']
        linear_size = network_params['discriminator_linear_size']
        linear_dropout = network_params['linear_dropout']
        
        image_size = training_params['image_size']
        
        self.model_version = training_params['model_version']
        
        # CNN
        self.cnn = nn.Sequential(
            self._conv_block(image_channels, conv_size, conv_dropout=conv_dropout),
            self._conv_block(conv_size, conv_size*2, conv_dropout=conv_dropout),
            self._conv_block(conv_size*2, conv_size*4, conv_dropout=conv_dropout),
            self._conv_block(conv_size*4, conv_size*4, conv_dropout=conv_dropout),
            
            nn.Flatten(),
        )
        
        # Classifier
        n_channels = self.cnn(torch.empty(1, 1, image_size, image_size)).size(-1)
        
        self.classifier = nn.Sequential(
            nn.Dropout(linear_dropout),
            nn.Linear(n_channels, linear_size),
            nn.ReLU(),
            nn.Dropout(linear_dropout),
            nn.Linear(linear_size, 1),
    )
        
    def _conv_block(self, image_channels, conv_size, conv_dropout=0.2, kernel_size=5, stride=1):
        # Convolutional block
        return nn.Sequential(
            nn.Conv2d(image_channels, conv_size, kernel_size=kernel_size, stride=stride), 
            nn.InstanceNorm2d(conv_size, affine=True), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(conv_dropout)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.classifier(x)
        
        if self.model_version=='CGAN':
            x = torch.sigmoid(x)
        
        return x

class Discriminator_v4(nn.Module):
    """
    Discriminator that uses layer norm and leaky ReLU
    """
    def __init__(self, **training_params):
        super().__init__()
        # Params
        network_params = training_params['network_params']
        
        image_channels = network_params['image_channels']
        conv_size = network_params['discriminator_conv_size']
        linear_size = network_params['discriminator_linear_size']
        linear_dropout = network_params['linear_dropout']

        image_size = training_params['image_size']
        
        self.model_version = training_params['model_version']
        
        # CNN
        self.cnn = nn.Sequential(
            *self._conv_block(image_channels, conv_size, image_size),
            *self._conv_block(conv_size, conv_size*2, self.norm_img_size),
            *self._conv_block(conv_size*2, conv_size*4, self.norm_img_size),
            *self._conv_block(conv_size*4, conv_size*4, self.norm_img_size),
            
            nn.Flatten(),
        )
        # Classifier
        n_channels = self.cnn(torch.empty(1, 1, image_size, image_size)).size(-1)
        
        self.classifier = nn.Sequential(
            nn.Dropout(linear_dropout),
            nn.Linear(n_channels, linear_size),
            nn.ReLU(),
            nn.Dropout(linear_dropout),
            nn.Linear(linear_size, 1),
    )
        
    def _conv_block(self, image_channels, conv_size, img_size, kernel_size=5, stride=1):
        # Convolutional block
        layers = [nn.Conv2d(image_channels, conv_size, kernel_size=kernel_size, stride=stride)]
        self.norm_img_size = int((img_size-kernel_size)/stride + 1)
        layers.append(nn.LayerNorm([conv_size, self.norm_img_size, self.norm_img_size]))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return layers

    def forward(self, x):
        x = self.cnn(x)
        x = self.classifier(x)
        
        if self.model_version=='CGAN':
            x = torch.sigmoid(x)
        
        return x
    
class Discriminator_v5(nn.Module):
    """
    Discriminator that uses layer norm and leaky ReLU
    """
    def __init__(self, **training_params):
        super().__init__()
        # Params
        network_params = training_params['network_params']
        
        image_channels = network_params['image_channels']
        conv_size = network_params['discriminator_conv_size']
        linear_size = network_params['discriminator_linear_size']
        linear_dropout = network_params['linear_dropout']
        
        image_size = training_params['image_size']
        
        self.model_version = training_params['model_version']
        
        # CNN
        self.cnn = nn.Sequential(
            *self._conv_block(image_channels, conv_size, image_size),
            *self._conv_block(conv_size, conv_size*2, self.norm_img_size),
            *self._conv_block(conv_size*2, conv_size*4, self.norm_img_size),
            *self._conv_block(conv_size*4, conv_size*8, self.norm_img_size),
            *self._conv_block(conv_size*8, conv_size*8, self.norm_img_size),
            
            nn.Flatten(),
        )
        # Classifier
        n_channels = self.cnn(torch.empty(1, 1, image_size, image_size)).size(-1)
        
        self.classifier = nn.Sequential(
            nn.Dropout(linear_dropout),
            nn.Linear(n_channels, linear_size),
            nn.ReLU(),
            nn.Dropout(linear_dropout),
            nn.Linear(linear_size, 1),
    )
        
    def _conv_block(self, image_channels, conv_size, img_size, kernel_size=5, stride=1):
        # Convolutional block
        layers = [nn.Conv2d(image_channels, conv_size, kernel_size=kernel_size, stride=stride)]
        self.norm_img_size = int((img_size-kernel_size)/stride + 1)
        layers.append(nn.LayerNorm([conv_size, self.norm_img_size, self.norm_img_size]))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return layers

    def forward(self, x):
        x = self.cnn(x)
        x = self.classifier(x)
        
        if self.model_version=='CGAN':
            x = torch.sigmoid(x)
        
        return x

class Discriminator_v6(nn.Module):
    """
    Discriminator that uses layer norm and leaky ReLU
    """
    def __init__(self, **training_params):
        super().__init__()
        # Params
        network_params = training_params['network_params']
        
        image_channels = network_params['image_channels']
        conv_size = network_params['discriminator_conv_size']
        linear_size = network_params['discriminator_linear_size']
        linear_dropout = network_params['linear_dropout']
        
        image_size = training_params['image_size']
        
        self.model_version = training_params['model_version']
        
        # CNN
        self.cnn = nn.Sequential(
            *self._conv_block(image_channels, conv_size, image_size),
            *self._conv_block(conv_size, conv_size*2, self.norm_img_size),
            *self._conv_block(conv_size*2, conv_size*4, self.norm_img_size),
            *self._conv_block(conv_size*4, conv_size*8, self.norm_img_size),
            
            nn.Flatten(),
        )
        # Classifier
        n_channels = self.cnn(torch.empty(1, 1, image_size, image_size)).size(-1)
        
        self.classifier = nn.Sequential(
            nn.Dropout(linear_dropout),
            nn.Linear(n_channels, linear_size),
            nn.ReLU(),
            nn.Dropout(linear_dropout),
            nn.Linear(linear_size, 1),
    )
        
    def _conv_block(self, image_channels, conv_size, img_size, kernel_size=5, stride=1):
        # Convolutional block
        layers = [nn.Conv2d(image_channels, conv_size, kernel_size=kernel_size, stride=stride)]
        self.norm_img_size = int((img_size-kernel_size)/stride + 1)
        layers.append(nn.LayerNorm([conv_size, self.norm_img_size, self.norm_img_size]))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return layers

    def forward(self, x):
        x = self.cnn(x)
        x = self.classifier(x)
        
        if self.model_version=='CGAN':
            x = torch.sigmoid(x)
        
        return x



"""Generators"""
class Generator_v1(nn.Module):
    """
    Simple generator that uses convTranspose to upsample
    """
    def __init__(self, **training_params):
        super().__init__()
        
        # Params
        network_params = training_params['network_params']
        
        latent_dim = network_params['latent_dim']
        upsamp_size = network_params['generator_upsamp_size']
        gen_img_w = network_params['generator_img_w']
        
        image_size = training_params['image_size']
        
        # Pass latent space input into linear layer and reshape
        self.linear = nn.Sequential(
            nn.Linear(latent_dim, gen_img_w*gen_img_w*upsamp_size),
            nn.ReLU(inplace=True),
            
            nn.Unflatten(1, (upsamp_size, gen_img_w, gen_img_w))
        )
        
        # Upsample
        final_kernel_size = int((gen_img_w*2+2)*2+2 - image_size + 1)
        
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(upsamp_size, int(upsamp_size/2), kernel_size=4, stride=2), 
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(int(upsamp_size/2), int(upsamp_size/4), kernel_size=4, stride=2), 
            nn.ReLU(inplace=True),
            
            nn.Conv2d(int(upsamp_size/4), 1, kernel_size=final_kernel_size)
        )

    def forward(self, x):
        x = self.linear(x)
        x = self.upsample(x)
        return x
    
class Generator_v2(nn.Module):
    """
    Simple generator that uses nearest neighbour to upsample
    """
    def __init__(self, **training_params):
        super().__init__()
        
        # Params
        network_params = training_params['network_params']
        
        latent_dim = network_params['latent_dim']
        upsamp_size = network_params['generator_upsamp_size']
        gen_img_w = network_params['generator_img_w']
        
        image_size = training_params['image_size']
        
        # Pass latent space input into linear layer and reshape
        self.linear = nn.Sequential(
            nn.Linear(latent_dim, gen_img_w*gen_img_w*upsamp_size),
            nn.ReLU(inplace=True),
            
            nn.Unflatten(1, (upsamp_size, gen_img_w, gen_img_w))
        )
        
        # Upsample
        final_kernel_size = int(gen_img_w*2*2 - image_size + 1)
        
        self.upsample = nn.Sequential(
            nn.Upsample(size=(gen_img_w*2, gen_img_w*2), mode='nearest'),
            nn.Conv2d(upsamp_size, int(upsamp_size/2), 3, 1),
            
            nn.Upsample(size=(gen_img_w*4, gen_img_w*4), mode='nearest'),
            nn.Conv2d(int(upsamp_size/2), int(upsamp_size/4), 3, 1),
            
            nn.Conv2d(int(upsamp_size/4), 1, kernel_size=final_kernel_size-2),
        )

    def forward(self, x):
        x = self.linear(x)
        x = self.upsample(x)
        return x
    
class Generator_v3(nn.Module):
    """
    Simple generator that uses nearest neighbour to upsample
    """
    def __init__(self, **training_params):
        super().__init__()
        
        # Params
        network_params = training_params['network_params']
        
        latent_dim = network_params['latent_dim']
        upsamp_size = network_params['generator_upsamp_size']
        gen_img_w = network_params['generator_img_w']
        
        image_size = training_params['image_size']
        
        # Pass latent space input into linear layer and reshape
        self.linear = nn.Sequential(
            nn.Linear(latent_dim, gen_img_w*gen_img_w*upsamp_size),
            nn.ReLU(inplace=True),
            
            nn.Unflatten(1, (upsamp_size, gen_img_w, gen_img_w))
        )
        
        # Upsample
        final_kernel_size = int(gen_img_w*2*2 - image_size + 1)
        
        self.upsample = nn.Sequential(
            nn.Upsample(size=(gen_img_w*2, gen_img_w*2), mode='nearest'),
            nn.Conv2d(upsamp_size, int(upsamp_size/2), 3, 1),
            
            nn.Upsample(size=(gen_img_w*4, gen_img_w*4), mode='nearest'),
            nn.Conv2d(int(upsamp_size/2), int(upsamp_size/4), 3, 1),
            
            nn.Conv2d(int(upsamp_size/4), 1, kernel_size=final_kernel_size-2),
        )

    def forward(self, x):
        x = self.linear(x)
        x = self.upsample(x)
        return x

class Generator_v4(nn.Module):
    """
    Generator that uses nearest neighbour to upsample
    """
    def __init__(self, **training_params):
        super().__init__()
        
        # Params
        network_params = training_params['network_params']
        
        latent_dim = network_params['latent_dim']
        upsamp_size = network_params['generator_upsamp_size']
        gen_img_w = network_params['generator_img_w']
        
        image_size = training_params['image_size']
        
        # Pass latent space input into linear layer and reshape
        self.linear = nn.Sequential(
            nn.Linear(latent_dim, gen_img_w*gen_img_w*upsamp_size),
            nn.ReLU(inplace=True),
            
            nn.Unflatten(1, (upsamp_size, gen_img_w, gen_img_w))
        )
        
        # Upsample
        final_kernel_size = int(gen_img_w*2*2*2 - image_size + 1)
        
        self.upsample = nn.Sequential(
            self._upsamp_block(gen_img_w, upsamp_size, int(upsamp_size/2)),
            self._upsamp_block(gen_img_w*2, int(upsamp_size/2), int(upsamp_size/4)),
            self._upsamp_block(gen_img_w*4, int(upsamp_size/4), 5),
            
            nn.Conv2d(5, 1, kernel_size=final_kernel_size-2),
        )
        
    def _upsamp_block(self, img_w, upsamp_size_input, upsamp_size_output):
        # Upsample block
        return nn.Sequential(
            nn.Upsample(size=(img_w*2, img_w*2), mode='nearest'),
            nn.Conv2d(upsamp_size_input, upsamp_size_output, 3, 1)
        )

    def forward(self, x):
        x = self.linear(x)
        x = self.upsample(x)
        return x

class Generator_v5(nn.Module):
    """
    Generator that uses nearest neighbour to upsample
    """
    def __init__(self, **training_params):
        super().__init__()
        
        # Params
        network_params = training_params['network_params']
        
        latent_dim = network_params['latent_dim']
        upsamp_size = network_params['generator_upsamp_size']
        gen_img_w = network_params['generator_img_w']
        
        image_size = training_params['image_size']
        
        # Pass latent space input into linear layer and reshape
        self.linear = nn.Sequential(
            nn.Linear(latent_dim, gen_img_w*gen_img_w*upsamp_size),
            nn.ReLU(inplace=True),
            
            nn.Unflatten(1, (upsamp_size, gen_img_w, gen_img_w))
        )
        
        # Upsample
        final_kernel_size = int(gen_img_w*2*2*2 - image_size + 1)
        
        self.upsample = nn.Sequential(
            self._upsamp_block(gen_img_w, upsamp_size, int(upsamp_size/2)),
            self._upsamp_block(gen_img_w*2, int(upsamp_size/2), int(upsamp_size/4)),
            self._upsamp_block(gen_img_w*4, int(upsamp_size/4), 5),
            
            nn.Conv2d(5, 1, kernel_size=final_kernel_size-2),
        )
        
    def _upsamp_block(self, img_w, upsamp_size_input, upsamp_size_output):
        # Upsample block
        return nn.Sequential(
            nn.Upsample(size=(img_w*2, img_w*2), mode='nearest'),
            nn.Conv2d(upsamp_size_input, upsamp_size_output, 3, 1),
            nn.BatchNorm2d(upsamp_size_output), nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        x = self.linear(x)
        x = self.upsample(x)
        return torch.tanh(x)
    
class Generator_v6(nn.Module):
    """
    Generator that uses nearest neighbour to upsample
    """
    def __init__(self, **training_params):
        super().__init__()
        
        # Params
        network_params = training_params['network_params']
        
        latent_dim = network_params['latent_dim']
        upsamp_size = network_params['generator_upsamp_size']
        gen_img_w = network_params['generator_img_w']
        
        image_size = training_params['image_size']
        
        # Pass latent space input into linear layer and reshape
        self.linear = nn.Sequential(
            nn.Linear(latent_dim, gen_img_w*gen_img_w*upsamp_size),
            nn.ReLU(inplace=True),
            
            nn.Unflatten(1, (upsamp_size, gen_img_w, gen_img_w))
        )
        
        # Upsample
        final_kernel_size = int(gen_img_w*(2**4) - image_size + 1)
        
        self.upsample = nn.Sequential(
            self._upsamp_block(gen_img_w, upsamp_size, int(upsamp_size/2)),
            self._upsamp_block(gen_img_w*2, int(upsamp_size/2), int(upsamp_size/4)),
            self._upsamp_block(gen_img_w*4, int(upsamp_size/4), int(upsamp_size/8)),
            self._upsamp_block(gen_img_w*8, int(upsamp_size/8), 5),
            
            nn.Conv2d(5, 1, kernel_size=final_kernel_size-2),
        )
        
    def _upsamp_block(self, img_w, upsamp_size_input, upsamp_size_output):
        # Upsample block
        return nn.Sequential(
            nn.Upsample(size=(img_w*2, img_w*2), mode='nearest'),
            nn.Conv2d(upsamp_size_input, upsamp_size_output, 3, 1),
            nn.BatchNorm2d(upsamp_size_output), nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        x = self.linear(x)
        x = self.upsample(x)
        return torch.tanh(x)
    
class Generator_v7(nn.Module):
    """
    Generator that uses nearest neighbour to upsample
    """
    def __init__(self, **training_params):
        super().__init__()
        
        # Params
        network_params = training_params['network_params']
        
        latent_dim = network_params['latent_dim']
        upsamp_size = network_params['generator_upsamp_size']
        gen_img_w = network_params['generator_img_w']
        
        image_size = training_params['image_size']
        
        # Pass latent space input into linear layer and reshape
        self.linear = nn.Sequential(
            nn.Linear(latent_dim, gen_img_w*gen_img_w*upsamp_size),
            nn.ReLU(inplace=True),
            
            nn.Unflatten(1, (upsamp_size, gen_img_w, gen_img_w))
        )
        
        # Upsample
        final_kernel_size = int(gen_img_w*(2**4) - image_size + 1)
        
        self.upsample = nn.Sequential(
            self._upsamp_block(gen_img_w, upsamp_size, int(upsamp_size/2)),
            self._upsamp_block(gen_img_w*2, int(upsamp_size/2), int(upsamp_size/4)),
            self._upsamp_block(gen_img_w*4, int(upsamp_size/4), int(upsamp_size/8)),
            self._upsamp_block(gen_img_w*8, int(upsamp_size/8), 10),
            
            nn.Conv2d(10, 1, kernel_size=final_kernel_size-2),
        )
        
    def _upsamp_block(self, img_w, upsamp_size_input, upsamp_size_output):
        # Upsample block
        return nn.Sequential(
            nn.Upsample(size=(img_w*2, img_w*2), mode='nearest'),
            nn.Conv2d(upsamp_size_input, upsamp_size_output, 3, 1),
            nn.BatchNorm2d(upsamp_size_output), nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        x = self.linear(x)
        x = self.upsample(x)
        return torch.tanh(x)



"""SAGAN"""
class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
        return out

class SaganGenerator(nn.Module):
    """Generator."""

    def __init__(self, **training_params):
        super(SaganGenerator, self).__init__()
        self.imsize = training_params['image_size']
        network_params = training_params['network_params']
        
        z_dim = network_params['latent_dim']
        image_channels = network_params['image_channels']
        layer_mults = network_params['gen_layer_mults']
        up_dim = network_params['gen_dim']
        gen_img_size = network_params['gen_img_size']

        dims = [*map(lambda m: up_dim * m, layer_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        
        # Latent space to 2D
        layer = []
        layer.append(spectral_norm(nn.Linear(z_dim, gen_img_size*gen_img_size*dims[0])))
        layer.append(nn.ReLU(inplace=True))
        layer.append(nn.Unflatten(image_channels, (dims[0], gen_img_size, gen_img_size)))
        
        self.linear = nn.Sequential(*layer)
        
        # Build up layers
        layers = []
        for i, (dim_in, dim_out) in enumerate(in_out[:-1]):
            layer = []
            layer.append(self._upsample_conv(gen_img_size * (2**i), dim_in, dim_out))
            layer.append(nn.BatchNorm2d(dim_out))
            layer.append(nn.LeakyReLU(0.2))
            
            layers.append(nn.Sequential(*layer))
            
            
        up_layers = nn.Sequential(*layers)

        # Place attention layer at the second last up layer
        attn = Self_Attn(dims[-2], 'relu')
        
        # Build last up layer
        dim_in, dim_out = in_out[-1]
        layer = []
        layer.append(self._upsample_conv(gen_img_size * (2**(len(dims)-1)), dim_in, dim_out))
        layer.append(nn.BatchNorm2d(dim_out))
        layer.append(nn.LeakyReLU(0.2))
        
        last_up = nn.Sequential(*layer)
        
        # All upsample layers
        self.upsample = nn.Sequential(up_layers, attn, last_up)
        
        # Resize image
        final_kernal_size = int(gen_img_size * (2**len(dims)) - self.imsize - 1)
        layer = []
        layer.append(spectral_norm(nn.Conv2d(dims[-1], 1, kernel_size=final_kernal_size)))
        
        self.resize = nn.Sequential(*layer)
        
    def _upsample_conv(self, img_w, in_dim, out_dim):
        """Upsampling block by nearest neighbour method"""
        layer = []
        layer.append(nn.Upsample(size=(img_w*2, img_w*2), mode='nearest'))
        layer.append(spectral_norm(nn.Conv2d(in_dim, out_dim, 3, 1)))
        return nn.Sequential(*layer)
            
    def forward(self, z):
        x = self.linear(z)
        x = self.upsample(x)
        out = self.resize(x)
        return torch.tanh(out)


class SaganDiscriminator(nn.Module):
    """Discriminator, Auxiliary Classifier."""

    def __init__(self, **training_params):
        super(SaganDiscriminator, self).__init__()
        self.imsize = training_params['image_size']
        network_params = training_params['network_params']
        self.model_version = training_params['model_version']
        
        image_channels = network_params['image_channels']
        layer_mults = network_params['dis_layer_mults']
        conv_dim = network_params['dis_dim']
        
        dims = [image_channels, *map(lambda m: conv_dim * m, layer_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # Build convolutional layers
        layers = []
        for (dim_in, dim_out) in in_out[:-1]:
            layer = []
            layer.append(spectral_norm(nn.Conv2d(dim_in, dim_out, 4, 2, 1)))
            layer.append(nn.LeakyReLU(0.2))
            
            layers.append(nn.Sequential(*layer))
        conv_layers = nn.Sequential(*layers)

        # Place attention at the second last conv layer
        attn = Self_Attn(dims[-2], 'relu')
        
        # Build last convolutional layer
        dim_in, dim_out = in_out[-1]
        layer = []
        layer.append(spectral_norm(nn.Conv2d(dim_in, dim_out, 4, 2, 1)))
        layer.append(nn.LeakyReLU(0.2))
        last_conv = nn.Sequential(*layer)
        
        # All cnn layers
        self.attn_cnn = nn.Sequential(conv_layers, attn, last_conv, nn.Flatten())
        
        # Classifier
        n_channels = self.attn_cnn(torch.empty(1, 1, self.imsize, self.imsize)).size(-1)

        layer = []
        layer.append(nn.Dropout(0.2))
        layer.append(spectral_norm(nn.Linear(n_channels, 1)))
        
        self.classifier = nn.Sequential(*layer)
        
    def forward(self, x):
        x = self.attn_cnn(x)
        out = self.classifier(x)
        
        if self.model_version=='CGAN':
            out = torch.sigmoid(out)
        
        return out
    
'Big GAN'
# Architectures for G
# Attention is passed in in the format '32_64' to mean applying an attention
# block at both resolution 32x32 and 64x64. Just '64' will apply at 64x64.
def G_arch(ch=64, attention='64', ksize='333333', dilation='111111'):
  arch = {}
  arch[512] = {'in_channels' :  [ch * item for item in [16, 16, 8, 8, 4, 2, 1]],
               'out_channels' : [ch * item for item in [16,  8, 8, 4, 2, 1, 1]],
               'upsample' : [True] * 7,
               'resolution' : [8, 16, 32, 64, 128, 256, 512],
               'attention' : {2**i: (2**i in [int(item) for item in attention.split('_')])
                              for i in range(3,10)}}
  arch[256] = {'in_channels' :  [ch * item for item in [16, 16, 8, 8, 4, 2]],
               'out_channels' : [ch * item for item in [16,  8, 8, 4, 2, 1]],
               'upsample' : [True] * 6,
               'resolution' : [8, 16, 32, 64, 128, 256],
               'attention' : {2**i: (2**i in [int(item) for item in attention.split('_')])
                              for i in range(3,9)}}
  arch[128] = {'in_channels' :  [ch * item for item in [16, 16, 8, 4, 2]],
               'out_channels' : [ch * item for item in [16, 8, 4, 2, 1]],
               'upsample' : [True] * 5,
               'resolution' : [8, 16, 32, 64, 128],
               'attention' : {2**i: (2**i in [int(item) for item in attention.split('_')])
                              for i in range(3,8)}}
  arch[64]  = {'in_channels' :  [ch * item for item in [16, 16, 8, 4]],
               'out_channels' : [ch * item for item in [16, 8, 4, 2]],
               'upsample' : [True] * 4,
               'resolution' : [8, 16, 32, 64],
               'attention' : {2**i: (2**i in [int(item) for item in attention.split('_')])
                              for i in range(3,7)}}
  arch[32]  = {'in_channels' :  [ch * item for item in [4, 4, 4]],
               'out_channels' : [ch * item for item in [4, 4, 4]],
               'upsample' : [True] * 3,
               'resolution' : [8, 16, 32],
               'attention' : {2**i: (2**i in [int(item) for item in attention.split('_')])
                              for i in range(3,6)}}

  return arch

# Discriminator architecture, same paradigm as G's above
def D_arch(ch=64, image_channels=1, attention='64',ksize='333333', dilation='111111'):
  arch = {}
  arch[256] = {'in_channels' :  [image_channels] + [ch*item for item in [1, 2, 4, 8, 8, 16]],
               'out_channels' : [item * ch for item in [1, 2, 4, 8, 8, 16, 16]],
               'downsample' : [True] * 6 + [False],
               'resolution' : [128, 64, 32, 16, 8, 4, 4 ],
               'attention' : {2**i: 2**i in [int(item) for item in attention.split('_')]
                              for i in range(2,8)}}
  arch[128] = {'in_channels' :  [image_channels] + [ch*item for item in [1, 2, 4, 8, 16]],
               'out_channels' : [item * ch for item in [1, 2, 4, 8, 16, 16]],
               'downsample' : [True] * 5 + [False],
               'resolution' : [64, 32, 16, 8, 4, 4],
               'attention' : {2**i: 2**i in [int(item) for item in attention.split('_')]
                              for i in range(2,8)}}
  arch[64]  = {'in_channels' :  [image_channels] + [ch*item for item in [1, 2, 4, 8]],
               'out_channels' : [item * ch for item in [1, 2, 4, 8, 16]],
               'downsample' : [True] * 4 + [False],
               'resolution' : [32, 16, 8, 4, 4],
               'attention' : {2**i: 2**i in [int(item) for item in attention.split('_')]
                              for i in range(2,7)}}
  arch[32]  = {'in_channels' :  [image_channels] + [item * ch for item in [4, 4, 4]],
               'out_channels' : [item * ch for item in [4, 4, 4, 4]],
               'downsample' : [True, True, True, False],
               'resolution' : [16, 8, 4, 4],
               'attention' : {2**i: 2**i in [int(item) for item in attention.split('_')]
                              for i in range(2,6)}}
  return arch


class BigGanGenerator(nn.Module):
  def __init__(self, **training_params):
    super(BigGanGenerator, self).__init__()
    network_params = training_params['network_params']
    # Number of channels
    image_channels = network_params['image_channels']
    # Channel width mulitplier
    self.ch = network_params['G_channels']
    # Dimensionality of the latent space
    self.dim_z = network_params['latent_dim']
    # The initial spatial dimensions
    self.bottom_width = network_params['bottom_width']
    # Resolution of the output
    self.resolution = training_params['image_size']
    # Kernel size?
    self.kernel_size = 3
    # Attention?
    self.attention = network_params['G_attention']
    # number of classes, for use in categorical conditional generation
    self.n_classes = None
    # Hierarchical latent space?
    self.hier = False
    # Cross replica batchnorm?
    self.cross_replica = False
    # Use my batchnorm?
    self.mybn = False
    # nonlinearity for residual blocks
    self.activation = nn.ReLU(inplace=False)
    # Initialization style
    self.init = 'ortho'
    # Parameterization style
    self.G_param = 'SN'
    # Normalization style
    self.norm_style = 'bn'
    # Epsilon for BatchNorm?
    self.BN_eps = 1e-5
    # Epsilon for Spectral Norm?
    self.SN_eps = 1e-12
    # fp16?
    self.fp16 = False
    # Architecture dict
    self.arch = G_arch(self.ch, attention=self.attention)[self.resolution]
    # Other params
    skip_init=False
    num_G_SVs=1
    num_G_SV_itrs=1

    # If using hierarchical latents, adjust z
    if self.hier:
      # Number of places z slots into
      self.num_slots = len(self.arch['in_channels']) + 1
      self.z_chunk_size = (self.dim_z // self.num_slots)
      # Recalculate latent dimensionality for even splitting into chunks
      self.dim_z = self.z_chunk_size *  self.num_slots
    else:
      self.num_slots = 1
      self.z_chunk_size = 0

    # Which convs, batchnorms, and linear layers to use
    if self.G_param == 'SN':
      self.which_conv = functools.partial(layers.SNConv2d,
                          kernel_size=3, padding=1,
                          num_svs=num_G_SVs, num_itrs=num_G_SV_itrs,
                          eps=self.SN_eps)
      self.which_linear = functools.partial(layers.SNLinear,
                          num_svs=num_G_SVs, num_itrs=num_G_SV_itrs,
                          eps=self.SN_eps)
    else:
      self.which_conv = functools.partial(nn.Conv2d, kernel_size=3, padding=1)
      self.which_linear = nn.Linear
      
    # We use a non-spectral-normed embedding here regardless;
    # For some reason applying SN to G's embedding seems to randomly cripple G
    self.which_embedding = nn.Embedding
    self.which_bn = functools.partial(layers.bn,
                          cross_replica=self.cross_replica,
                          mybn=self.mybn,
                          eps=self.BN_eps)


    # Prepare model
    # If not using shared embeddings, self.shared is just a passthrough
    # self.shared = (self.which_embedding(n_classes, self.shared_dim) if G_shared 
                    # else layers.identity())
    # First linear layer
    self.linear = self.which_linear(self.dim_z // self.num_slots,
                                    self.arch['in_channels'][0] * (self.bottom_width **2))

    # self.blocks is a doubly-nested list of modules, the outer loop intended
    # to be over blocks at a given resolution (resblocks and/or self-attention)
    # while the inner loop is over a given block
    self.blocks = []
    for index in range(len(self.arch['out_channels'])):
      self.blocks += [[layers.GBlock(in_channels=self.arch['in_channels'][index],
                             out_channels=self.arch['out_channels'][index],
                             which_conv=self.which_conv,
                             which_bn=self.which_bn,
                             activation=self.activation,
                             upsample=(functools.partial(F.interpolate, scale_factor=2)
                                       if self.arch['upsample'][index] else None))]]

      # If attention on this block, attach it to the end
      if self.arch['attention'][self.arch['resolution'][index]]:
        print('Adding attention layer in G at resolution %d' % self.arch['resolution'][index])
        self.blocks[-1] += [layers.Attention(self.arch['out_channels'][index], self.which_conv)]

    # Turn self.blocks into a ModuleList so that it's all properly registered.
    self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])

    # output layer: batchnorm-relu-conv.
    # Consider using a non-spectral conv here
    self.output_layer = nn.Sequential(layers.bn(self.arch['out_channels'][-1],
                                                cross_replica=self.cross_replica,
                                                mybn=self.mybn),
                                    self.activation,
                                    self.which_conv(self.arch['out_channels'][-1], image_channels))

    # Initialize weights. Optionally skip init for testing.
    if not skip_init:
      self.init_weights()

    '''
    # Set up optimizer
    # If this is an EMA copy, no need for an optim, so just return now
    if no_optim:
      return
    self.lr, self.B1, self.B2, self.adam_eps = G_lr, G_B1, G_B2, adam_eps
    if G_mixed_precision:
      print('Using fp16 adam in G...')
      import utilsi
      self.optim = utils.Adam16(params=self.parameters(), lr=self.lr,
                           betas=(self.B1, self.B2), weight_decay=0,
                           eps=self.adam_eps)
    else:
      self.optim = optim.Adam(params=self.parameters(), lr=self.lr,
                           betas=(self.B1, self.B2), weight_decay=0,
                           eps=self.adam_eps)
    '''

    # LR scheduling, left here for forward compatibility
    # self.lr_sched = {'itr' : 0}# if self.progressive else {}
    # self.j = 0

  # Initialize
  def init_weights(self):
    self.param_count = 0
    for module in self.modules():
      if (isinstance(module, nn.Conv2d) 
          or isinstance(module, nn.Linear) 
          or isinstance(module, nn.Embedding)):
        if self.init == 'ortho':
          init.orthogonal_(module.weight)
        elif self.init == 'N02':
          init.normal_(module.weight, 0, 0.02)
        elif self.init in ['glorot', 'xavier']:
          init.xavier_uniform_(module.weight)
        else:
          print('Init style not recognized...')
        self.param_count += sum([p.data.nelement() for p in module.parameters()])
    print('Param count for G''s initialized parameters: %d' % self.param_count)

  # Note on this forward function: we pass in a y vector which has
  # already been passed through G.shared to enable easy class-wise
  # interpolation later. If we passed in the one-hot and then ran it through
  # G.shared in this forward function, it would be harder to handle.
  def forward(self, z, y=None):
    # If hierarchical, concatenate zs and ys
    if self.hier:
      zs = torch.split(z, self.z_chunk_size, 1)
      z = zs[0]
      ys = [torch.cat([y, item], 1) for item in zs[1:]]
    else:
      ys = [y] * len(self.blocks)
      
    # First linear layer
    h = self.linear(z)
    # Reshape
    h = h.view(h.size(0), -1, self.bottom_width, self.bottom_width)
    
    # Loop over blocks
    for index, blocklist in enumerate(self.blocks):
      # Second inner loop in case block has multiple layers
      for block in blocklist:
        h = block(h, ys[index])
        
    # Apply batchnorm-relu-conv-tanh at output
    out = torch.tanh(self.output_layer(h))
    return out

class BigGanDiscriminator(nn.Module):
  def __init__(self, **training_params):
    super(BigGanDiscriminator, self).__init__()
    network_params = training_params['network_params']
    # Image channels
    image_channels = network_params['image_channels']
    # Width mulitplier
    self.ch = network_params['D_channels']
    # Resolution of the output
    self.resolution = training_params['image_size']
    # Use Wide D as in BigGAN and SA-GAN or skinny D as in SN-GAN?
    self.D_wide = True
    # Kernel size
    self.kernel_size = 3
    # Attention?
    self.attention = network_params['D_attention']
    # Number of classes
    self.n_classes = None
    # Activation
    self.activation = nn.ReLU(inplace=False)
    # Initialization style
    self.init = 'ortho'
    # Parameterization style
    self.D_param = 'SN'
    # Epsilon for Spectral Norm?
    self.SN_eps = 1e-12
    # Fp16?
    self.fp16 = False
    # Architecture
    self.arch = D_arch(self.ch, attention=self.attention, image_channels=image_channels)[self.resolution]
    # Other params
    output_dim=1
    skip_init = False
    num_D_SVs=1
    num_D_SV_itrs=1

    # Which convs, batchnorms, and linear layers to use
    # No option to turn off SN in D right now
    if self.D_param == 'SN':
      self.which_conv = functools.partial(layers.SNConv2d,
                          kernel_size=3, padding=1,
                          num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                          eps=self.SN_eps)
      self.which_linear = functools.partial(layers.SNLinear,
                          num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                          eps=self.SN_eps)
      self.which_embedding = functools.partial(layers.SNEmbedding,
                              num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                              eps=self.SN_eps)
    # Prepare model
    # self.blocks is a doubly-nested list of modules, the outer loop intended
    # to be over blocks at a given resolution (resblocks and/or self-attention)
    self.blocks = []
    for index in range(len(self.arch['out_channels'])):
      self.blocks += [[layers.DBlock(in_channels=self.arch['in_channels'][index],
                       out_channels=self.arch['out_channels'][index],
                       which_conv=self.which_conv,
                       wide=self.D_wide,
                       activation=self.activation,
                       preactivation=(index > 0),
                       downsample=(nn.AvgPool2d(2) if self.arch['downsample'][index] else None))]]
      # If attention on this block, attach it to the end
      if self.arch['attention'][self.arch['resolution'][index]]:
        print('Adding attention layer in D at resolution %d' % self.arch['resolution'][index])
        self.blocks[-1] += [layers.Attention(self.arch['out_channels'][index],
                                             self.which_conv)]
    # Turn self.blocks into a ModuleList so that it's all properly registered.
    self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])
    # Linear output layer. The output dimension is typically 1, but may be
    # larger if we're e.g. turning this into a VAE with an inference output
    self.linear = self.which_linear(self.arch['out_channels'][-1], output_dim)
    ## UNCONDITIONAL, embed not needed ##
    # Embedding for projection discrimination
    # self.embed = self.which_embedding(self.n_classes, self.arch['out_channels'][-1])

    # Initialize weights
    if not skip_init:
      self.init_weights()
   
    '''
    # Set up optimizer
    self.lr, self.B1, self.B2, self.adam_eps = D_lr, D_B1, D_B2, adam_eps
    if D_mixed_precision:
      print('Using fp16 adam in D...')
      import utils
      self.optim = utils.Adam16(params=self.parameters(), lr=self.lr,
                             betas=(self.B1, self.B2), weight_decay=0, eps=self.adam_eps)
    else:
      self.optim = optim.Adam(params=self.parameters(), lr=self.lr,
                             betas=(self.B1, self.B2), weight_decay=0, eps=self.adam_eps)
    # LR scheduling, left here for forward compatibility
    # self.lr_sched = {'itr' : 0}# if self.progressive else {}
    # self.j = 0
    '''

  # Initialize
  def init_weights(self):
    self.param_count = 0
    for module in self.modules():
      if (isinstance(module, nn.Conv2d)
          or isinstance(module, nn.Linear)
          or isinstance(module, nn.Embedding)):
        if self.init == 'ortho':
          init.orthogonal_(module.weight)
        elif self.init == 'N02':
          init.normal_(module.weight, 0, 0.02)
        elif self.init in ['glorot', 'xavier']:
          init.xavier_uniform_(module.weight)
        else:
          print('Init style not recognized...')
        self.param_count += sum([p.data.nelement() for p in module.parameters()])
    print('Param count for D''s initialized parameters: %d' % self.param_count)

  def forward(self, x, y=None):
    # Stick x into h for cleaner for loops without flow control
    h = x
    # Loop over blocks
    for index, blocklist in enumerate(self.blocks):
      for block in blocklist:
        h = block(h)
    # Apply global sum pooling as in SN-GAN
    h = torch.sum(self.activation(h), [2, 3])
    # Get initial class-unconditional output
    out = self.linear(h)
    ## UNCONDITIONAL, step not needed ##
    # Get projection of final featureset onto class vectors and add to evidence
    # out = out + torch.sum(self.embed(y) * h, 1, keepdim=True) 
    return out

"""Diffusion"""

'Unet 1'
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels//2)
        )
        
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels),
        )
    
    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:,:,None,None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb
    
class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )
    
    def forward(self, x):
        x = x.view(-1, self.channels, self.size*self.size).swapaxes(1,2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )
        
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels),
        )
    
    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:,:,None,None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb
    
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )
    
    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else: 
            return self.double_conv(x)

class UNet_v1(nn.Module):
    def __init__(self, **training_params):
        super().__init__()
        
        network_params = training_params['network_params']
        image_size = training_params['image_size']
        
        self.time_dim = network_params['time_dim']
        
        image_channels = network_params['image_channels']
        initial_size = network_params['initial_size']
        
        bot_size = initial_size * 8
        
        self.inc = DoubleConv(image_channels, initial_size)
        self.down1 = Down(initial_size, initial_size*2, emb_dim=self.time_dim)
        self.sa1 = SelfAttention(initial_size*2, int(image_size/2))
        self.down2 = Down(initial_size*2, initial_size*4, emb_dim=self.time_dim)
        self.sa2 = SelfAttention(initial_size*4, int(image_size/4))
        self.down3 = Down(initial_size*4, initial_size*4, emb_dim=self.time_dim)
        self.sa3 = SelfAttention(initial_size*4, int(image_size/8))        

        self.bot1 = DoubleConv(initial_size*4, bot_size)
        self.bot2 = DoubleConv(bot_size, bot_size)
        self.bot3 = DoubleConv(bot_size, initial_size*4)
        
        self.up1 = Up(initial_size*8, initial_size*2, emb_dim=self.time_dim)
        self.sa4 = SelfAttention(initial_size*2, int(image_size/4))
        self.up2 = Up(initial_size*4, initial_size, emb_dim=self.time_dim)
        self.sa5 = SelfAttention(initial_size, int(image_size/2))
        self.up3 = Up(initial_size*2, initial_size, emb_dim=self.time_dim)
        self.sa6 = SelfAttention(initial_size, image_size)
        self.outc = nn.Conv2d(initial_size, image_channels, kernel_size=1)
        
    def pos_encoding(self, t, channels):
        device = t.device
        inv_freq = 1. / (
            10000**(torch.arange(0, channels, 2, device=device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels//2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels//2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc
    
    def forward(self, x, t):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)
        
        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)
        
        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)
        
        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)
        
        return output
        
'unet v2'
# Main Model
class UnetConvNextBlock(nn.Module):
    def __init__(
        self,
        with_time_emb = True,
        output_mean_scale = False,
        residual = False,
        **training_params
    ):
        super().__init__()
        network_params = training_params['network_params']

        try:
            dim = network_params['model_dim']
            time_dim = network_params['time_dim']
        except KeyError:
            dim = training_params['image_size']
            time_dim = dim
        self.channels = network_params['image_channels']
        dim_mults = network_params['dim_mult']
        out_dim = self.channels
         
        self.residual = residual
        self.output_mean_scale = output_mean_scale

        dims = [self.channels, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        if with_time_emb:
            self.time_mlp = nn.Sequential(
                layers.SinusoidalPosEmb(time_dim),
                nn.Linear(time_dim, time_dim * 4),
                nn.GELU(),
                nn.Linear(time_dim * 4, time_dim) # need to check if this should be all time_dim
            )
        else:
            time_dim = None
            self.time_mlp = None

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                layers.ConvNextBlock(dim_in, dim_out, time_emb_dim = time_dim, norm = ind != 0),
                layers.ConvNextBlock(dim_out, dim_out, time_emb_dim = time_dim),
                layers.Residual(layers.PreNorm(dim_out, layers.LinearAttention(dim_out))),
                layers.Downsample(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = layers.ConvNextBlock(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn = layers.Residual(layers.PreNorm(mid_dim, layers.LinearAttention(mid_dim)))
        self.mid_block2 = layers.ConvNextBlock(mid_dim, mid_dim, time_emb_dim = time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                layers.ConvNextBlock(dim_out * 2, dim_in, time_emb_dim = time_dim),
                layers.ConvNextBlock(dim_in, dim_in, time_emb_dim = time_dim),
                layers.Residual(layers.PreNorm(dim_in, layers.LinearAttention(dim_in))),
                layers.Upsample(dim_in) if not is_last else nn.Identity()
            ]))

        out_dim = layers.default(out_dim, self.channels)
        self.final_conv = nn.Sequential(
            layers.ConvNextBlock(dim, dim),
            nn.Conv2d(dim, out_dim, 1),
            #nn.Tanh() # ADDED
        )

    def forward(self, x, time=None):
        orig_x = x
        t = None
        if time is not None and layers.exists(self.time_mlp):
            t = self.time_mlp(time)
        
        original_mean = torch.mean(x, [1, 2, 3], keepdim=True)
        h = []

        for convnext, convnext2, attn, downsample in self.downs:
            x = convnext(x, t)
            x = convnext2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for convnext, convnext2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = convnext(x, t)
            x = convnext2(x, t)
            x = attn(x)
            x = upsample(x)
        if self.residual:
            return self.final_conv(x) + orig_x

        out = self.final_conv(x)
        if self.output_mean_scale:
            out_mean = torch.mean(out, [1,2,3], keepdim=True)
            out = out - original_mean + out_mean

        return out
    
"""Dictionary for all networks"""
network_dict = {
    'gen_v1': Generator_v1,
    'gen_v2': Generator_v2,
    'gen_v3': Generator_v3,
    'gen_v4': Generator_v4,
    'gen_v5': Generator_v5,
    'gen_v6': Generator_v6,
    'gen_v7': Generator_v7,
    'dis_v1': Discriminator_v1,
    'dis_v2': Discriminator_v2,
    'dis_v3': Discriminator_v3,
    'dis_v4': Discriminator_v4,
    'dis_v5': Discriminator_v5,
    'dis_v6': Discriminator_v6,
    'dis_v7': SaganDiscriminator,
    'gen_v8': SaganGenerator,
    'dis_v8': BigGanDiscriminator,
    'gen_v9': BigGanGenerator,
    'unet_v1': UNet_v1,
    'unet_v2': UnetConvNextBlock
}