"""
Author: Nathan Teo

"""

import torch
import torch.nn as nn

"""Discriminators"""
# Detective: fake or no fake -> 1 output [0, 1]
class Discriminator_v1(nn.Module):
    """
    Simple discriminator based on standard convolutional neural network architechture
    """
    def __init__(self, **training_params):
        super().__init__()
        # Params
        input_channels = training_params['input_channels']
        conv_size = training_params['discriminator_conv_size']
        conv_dropout = training_params['conv_dropout']
        linear_size = training_params['discriminator_linear_size']
        linear_dropout = training_params['linear_dropout']
        image_size = training_params['image_size']
        
        self.gan_version = training_params['gan_version']
        
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
        
        if self.gan_version=='CGAN':
            x = torch.sigmoid(x)
            
        return x

class Discriminator_v2(nn.Module):
    """
    Simple discriminator that uses instance norm and leaky ReLU
    """
    def __init__(self, **training_params):
        super().__init__()
        # Params
        input_channels = training_params['input_channels']
        conv_size = training_params['discriminator_conv_size']
        conv_dropout = training_params['conv_dropout']
        linear_size = training_params['discriminator_linear_size']
        linear_dropout = training_params['linear_dropout']
        image_size = training_params['image_size']
        
        self.gan_version = training_params['gan_version']
        
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
        
        if self.gan_version=='CGAN':
            x = torch.sigmoid(x)
        
        return x
    
class Discriminator_v3(nn.Module):
    """
    Simple discriminator that uses instance norm and leaky ReLU
    """
    def __init__(self, **training_params):
        super().__init__()
        # Params
        input_channels = training_params['input_channels']
        conv_size = training_params['discriminator_conv_size']
        conv_dropout = training_params['conv_dropout']
        linear_size = training_params['discriminator_linear_size']
        linear_dropout = training_params['linear_dropout']
        image_size = training_params['image_size']
        
        self.gan_version = training_params['gan_version']
        
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
        
        if self.gan_version=='CGAN':
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
        latent_dim = training_params['latent_dim']
        upsamp_size = training_params['generator_upsamp_size']
        image_size = training_params['image_size']
        gen_img_w = training_params['generator_img_w']
        
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
        latent_dim = training_params['latent_dim']
        upsamp_size = training_params['generator_upsamp_size']
        image_size = training_params['image_size']
        gen_img_w = training_params['generator_img_w']
        
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
    def __init__(self, **training_params):
        super().__init__()
        
        # Params
        latent_dim = training_params['latent_dim']
        upsamp_size = training_params['generator_upsamp_size']
        image_size = training_params['image_size']
        gen_img_w = training_params['generator_img_w']
        
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
    def __init__(self, **training_params):
        super().__init__()
        
        # Params
        latent_dim = training_params['latent_dim']
        upsamp_size = training_params['generator_upsamp_size']
        image_size = training_params['image_size']
        gen_img_w = training_params['generator_img_w']
        
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
        )


    def forward(self, x):
        x = self.linear(x)
        x = self.upsample(x)
        return x


"""All models"""
models = {
    'gen_v1': Generator_v1,
    'gen_v2': Generator_v2,
    'gen_v3': Generator_v3,
    'gen_v4': Generator_v4,
    'dis_v1': Discriminator_v1,
    'dis_v2': Discriminator_v2,
    'dis_v3': Discriminator_v3,
}