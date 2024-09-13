"""
Author: Nathan Teo

This script contains generators and discriminators (or critics) used in the GANs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

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

"""Diffusion"""
'Blocks'
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

'UNet'
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
        inv_freq = 1. / (
            10000**(torch.arange(0, channels, 2).float() / channels)
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
    'unet_v1': UNet_v1
}