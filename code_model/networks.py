"""
Author: Nathan Teo

This script contains generators and discriminators (or critics) used in the GANs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from inspect import isfunction
from einops import rearrange


"""Discriminators"""
class Discriminator_v1(nn.Module):
    """
    Simple discriminator based on standard convolutional neural network architechture
    """
    def __init__(self, **training_params):
        super().__init__()
        # Params
        network_params = training_params['network_params']
        
        input_channels = network_params['input_channels']
        conv_size = network_params['discriminator_conv_size']
        conv_dropout = network_params['conv_dropout']
        linear_size = network_params['discriminator_linear_size']
        linear_dropout = network_params['linear_dropout']
        
        image_size = training_params['image_size']
        self.model_version = training_params['model_version']
        
        # Simple CNN
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, conv_size, kernel_size=5, stride=1), 
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
        
        input_channels = network_params['input_channels']
        conv_size = network_params['discriminator_conv_size']
        conv_dropout = network_params['conv_dropout']
        linear_size = network_params['discriminator_linear_size']
        linear_dropout = network_params['linear_dropout']

        image_size = training_params['image_size']
        
        self.model_version = training_params['model_version']
        
        # Simple CNN
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, conv_size, kernel_size=5, stride=1), 
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
        
        input_channels = network_params['input_channels']
        conv_size = network_params['discriminator_conv_size']
        conv_dropout = network_params['conv_dropout']
        linear_size = network_params['discriminator_linear_size']
        linear_dropout = network_params['linear_dropout']
        
        image_size = training_params['image_size']
        
        self.model_version = training_params['model_version']
        
        # CNN
        self.cnn = nn.Sequential(
            self._conv_block(input_channels, conv_size, conv_dropout=conv_dropout),
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
        
    def _conv_block(self, input_channels, conv_size, conv_dropout=0.2, kernel_size=5, stride=1):
        # Convolutional block
        return nn.Sequential(
            nn.Conv2d(input_channels, conv_size, kernel_size=kernel_size, stride=stride), 
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
        
        input_channels = network_params['input_channels']
        conv_size = network_params['discriminator_conv_size']
        linear_size = network_params['discriminator_linear_size']
        linear_dropout = network_params['linear_dropout']

        image_size = training_params['image_size']
        
        self.model_version = training_params['model_version']
        
        # CNN
        self.cnn = nn.Sequential(
            *self._conv_block(input_channels, conv_size, image_size),
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
        
    def _conv_block(self, input_channels, conv_size, img_size, kernel_size=5, stride=1):
        # Convolutional block
        layers = [nn.Conv2d(input_channels, conv_size, kernel_size=kernel_size, stride=stride)]
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
        
        input_channels = network_params['input_channels']
        conv_size = network_params['discriminator_conv_size']
        linear_size = network_params['discriminator_linear_size']
        linear_dropout = network_params['linear_dropout']
        
        image_size = training_params['image_size']
        
        self.model_version = training_params['model_version']
        
        # CNN
        self.cnn = nn.Sequential(
            *self._conv_block(input_channels, conv_size, image_size),
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
        
    def _conv_block(self, input_channels, conv_size, img_size, kernel_size=5, stride=1):
        # Convolutional block
        layers = [nn.Conv2d(input_channels, conv_size, kernel_size=kernel_size, stride=stride)]
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
        
        input_channels = network_params['input_channels']
        conv_size = network_params['discriminator_conv_size']
        linear_size = network_params['discriminator_linear_size']
        linear_dropout = network_params['linear_dropout']
        
        image_size = training_params['image_size']
        
        self.model_version = training_params['model_version']
        
        # CNN
        self.cnn = nn.Sequential(
            *self._conv_block(input_channels, conv_size, image_size),
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
        
    def _conv_block(self, input_channels, conv_size, img_size, kernel_size=5, stride=1):
        # Convolutional block
        layers = [nn.Conv2d(input_channels, conv_size, kernel_size=kernel_size, stride=stride)]
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
def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)

class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = nn.Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)

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

    def __init__(self, batch_size, image_size=64, z_dim=100, conv_dim=64):
        super(SaganGenerator, self).__init__()
        self.imsize = image_size
        layer1 = []
        layer2 = []
        layer3 = []
        last = []

        repeat_num = int(np.log2(self.imsize)) - 3
        mult = 2 ** repeat_num # 8
        layer1.append(SpectralNorm(nn.ConvTranspose2d(z_dim, conv_dim * mult, 4)))
        layer1.append(nn.BatchNorm2d(conv_dim * mult))
        layer1.append(nn.ReLU())

        curr_dim = conv_dim * mult

        layer2.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1)))
        layer2.append(nn.BatchNorm2d(int(curr_dim / 2)))
        layer2.append(nn.ReLU())

        curr_dim = int(curr_dim / 2)

        layer3.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1)))
        layer3.append(nn.BatchNorm2d(int(curr_dim / 2)))
        layer3.append(nn.ReLU())

        if self.imsize == 64:
            layer4 = []
            curr_dim = int(curr_dim / 2)
            layer4.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1)))
            layer4.append(nn.BatchNorm2d(int(curr_dim / 2)))
            layer4.append(nn.ReLU())
            self.l4 = nn.Sequential(*layer4)
            curr_dim = int(curr_dim / 2)

        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)

        last.append(nn.ConvTranspose2d(curr_dim, 3, 4, 2, 1))
        last.append(nn.Tanh())
        self.last = nn.Sequential(*last)

        self.attn1 = Self_Attn( 128, 'relu')
        self.attn2 = Self_Attn( 64,  'relu')

    def forward(self, z):
        z = z.view(z.size(0), z.size(1), 1, 1)
        out=self.l1(z)
        out=self.l2(out)
        out=self.l3(out)
        out,p1 = self.attn1(out)
        out=self.l4(out)
        out,p2 = self.attn2(out)
        out=self.last(out)

        return out, p1, p2


class SaganDiscriminator(nn.Module):
    """Discriminator, Auxiliary Classifier."""

    def __init__(self, **training_params):
        super(SaganDiscriminator, self).__init__()
        self.imsize = training_params['image_size']
        network_params = training_params['network_params']
        
        input_channels = network_params['input_channels']
        layer_mults = network_params['dis_layer_mults']
        conv_dim = network_params['conv_dim']
        
        # layer1 = []
        # layer2 = []
        # layer3 = []
        # last = []
        
        dims = [input_channels, *map(lambda m: conv_dim * m, layer_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        self.layers = []
        for (dim_in, dim_out) in in_out:
            layer = []
            layer.append(SpectralNorm(nn.Conv2d(dim_in, dim_out, 4, 2, 1)))
            layer.append(nn.LeakyReLU(0.1))
            
            self.layers.append(nn.Sequential(*layer))
        
        self.attn = Self_Attn(dims[-1], 'relu')
        
        self.last = nn.Sequential(nn.Conv2d(dims[-1], input_channels, 1, 4))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            
        x = self.attn(x)
        out = self.last(x)
        
        return out.squeeze()

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
        
        input_channels = network_params['input_channels']
        initial_size = network_params['initial_size']
        
        bot_size = initial_size * 8
        
        self.inc = DoubleConv(input_channels, initial_size)
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
        self.outc = nn.Conv2d(initial_size, input_channels, kernel_size=1)
        
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
        
'unet 2 from https://github.com/mikonvergence/DiffusionFastForward.git'
def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

class SinusoidalPosEmb(nn.Module):
    """
        Based on transformer-like embedding from 'Attention is all you need'
        Note: 10,000 corresponds to the maximum sequence length
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

def Upsample(dim):
    return nn.ConvTranspose2d(dim, dim, 4, 2, 1)

def Downsample(dim):
    return nn.Conv2d(dim, dim, 4, 2, 1)

class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

# building block modules
class ConvNextBlock(nn.Module):
    """ https://arxiv.org/abs/2201.03545 """

    def __init__(self, dim, dim_out, *, time_emb_dim = None, mult = 2, norm = True):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.GELU(),
            nn.Linear(time_emb_dim, dim)
        ) if exists(time_emb_dim) else None

        self.ds_conv = nn.Conv2d(dim, dim, 7, padding = 3, groups = dim)

        self.net = nn.Sequential(
            LayerNorm(dim) if norm else nn.Identity(),
            nn.Conv2d(dim, dim_out * mult, 3, padding = 1),
            nn.GELU(),
            nn.Conv2d(dim_out * mult, dim_out, 3, padding = 1)
        )

        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):
        h = self.ds_conv(x)

        if exists(self.mlp):
            assert exists(time_emb), 'time emb must be passed in'
            condition = self.mlp(time_emb)
            h = h + rearrange(condition, 'b c -> b c 1 1')

        h = self.net(h)
        return h + self.res_conv(x)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)
        q = q * self.scale

        k = k.softmax(dim = -1)
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)

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

        dim = network_params['model_dim']
        self.channels = network_params['input_channels']
        dim_mults = network_params['dim_mult']
        time_dim = network_params['time_dim']
        out_dim = self.channels
         
        self.residual = residual
        self.output_mean_scale = output_mean_scale

        dims = [self.channels, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        if with_time_emb:
            self.time_mlp = nn.Sequential(
                SinusoidalPosEmb(time_dim),
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
                ConvNextBlock(dim_in, dim_out, time_emb_dim = time_dim, norm = ind != 0),
                ConvNextBlock(dim_out, dim_out, time_emb_dim = time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = ConvNextBlock(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
        self.mid_block2 = ConvNextBlock(mid_dim, mid_dim, time_emb_dim = time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                ConvNextBlock(dim_out * 2, dim_in, time_emb_dim = time_dim),
                ConvNextBlock(dim_in, dim_in, time_emb_dim = time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))

        out_dim = default(out_dim, self.channels)
        self.final_conv = nn.Sequential(
            ConvNextBlock(dim, dim),
            nn.Conv2d(dim, out_dim, 1),
            #nn.Tanh() # ADDED
        )

    def forward(self, x, time=None):
        orig_x = x
        t = None
        if time is not None and exists(self.time_mlp):
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
    'unet_v1': UNet_v1,
    'unet_v2': UnetConvNextBlock
}