"""
Author: Nathan Teo

"""

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import wandb

import code_model.models as models
from code_model.plotting_utils import *

class GAN_utils():
    def __init__(self):
        pass
    
    def plot_imgs(self, gen_sample_imgs, real_sample_imgs):
        real_disc_scores = self.discriminator(real_sample_imgs).cpu().detach().numpy()[:,0]
        gen_disc_scores = self.discriminator(gen_sample_imgs).cpu().detach().numpy()[:,0]
        
        real_sample_imgs = real_sample_imgs.cpu().detach()[:,0,:,:]
        gen_sample_imgs = gen_sample_imgs.cpu().detach()[:,0,:,:]
        
        # Plotting grid of images
        fig = plt.figure(figsize=(8,5))
        subfig = fig.subfigures(1, 2, wspace=0.2)
        
        plot_img_grid(
            subfig[0], real_sample_imgs, 3, 
            title='Real Imgs', subplot_titles=real_disc_scores, wspace=.1, hspace=.1
            )
        plot_img_grid(
            subfig[1], gen_sample_imgs, 3,
            title='Generated Imgs', subplot_titles=gen_disc_scores, wspace=.1, hspace=.1
            )
        
        fig.suptitle(f'Epoch {self.current_epoch}')
        fig.text(.5, .03, 'disc score labelled above image', ha='center')
        plt.tight_layout()
        
        # Save
        plt.savefig(f'{self.root_path}/logs/{self.log_folder}/images/image_epoch{self.current_epoch}.png')
        plt.close('all')
    
    def log_losses(self, epoch_g_losses, epoch_d_losses):
        filename = f'{self.root_path}/logs/{self.log_folder}/losses.npz'
        
        if self.current_epoch == 0:
            epochs = [0] # Current epoch is 0
            g_losses = [np.mean(epoch_g_losses)]
            d_losses = [np.mean(epoch_d_losses)]
            np.savez(filename, epochs=epochs, g_losses=g_losses, d_losses=d_losses)
        else:
            f = np.load(filename, allow_pickle=True)
            epochs = np.append(f['epochs'], self.current_epoch)
            g_losses = np.append(f['g_losses'], np.mean(epoch_g_losses))
            d_losses = np.append(f['d_losses'], np.mean(epoch_d_losses))
            np.savez(filename, epochs=epochs, g_losses=g_losses, d_losses=d_losses)
        return [], []
        

class CGAN(pl.LightningModule, GAN_utils):
    def __init__(self, **training_params):
        super().__init__()
        self.automatic_optimization = False
        
        # Initialize params
        self.latent_dim = training_params['latent_dim']
        self.lr = training_params['lr']
        self.root_path = training_params['root_path']
        self.epoch_start_g_train = training_params['epoch_start_g_train']
        self.discriminator_train_freq = training_params['discriminator_train_freq']
        self.log_folder = training_params['model_name']
        
        gen_version = training_params['generator_version']
        dis_version = training_params['discriminator_version']
        
        self.epoch_d_losses = []
        self.epoch_g_losses = []
        
        # Initialize models
        self.generator = models.models[f'gen_v{gen_version}'](**training_params)
        self.discriminator = models.models[f'dis_v{dis_version}'](**training_params)

        # Random noise
        self.validation_z = torch.randn(9, training_params['latent_dim'])
        
    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)
    
    def training_step(self, batch, batch_idx):
        # Load real imgs
        if len(batch) == 2: # if label exists eg. MNIST dataset
            real_imgs, _ = batch
        else:
            real_imgs = batch
        
        # Log real imgs
        if batch_idx==0:
            sample_imgs = real_imgs[:9]
            grid = torchvision.utils.make_grid(sample_imgs)
            wandb.log({"real_images": wandb.Image(grid, caption="real_images")})
            self.real_sample_imgs = sample_imgs
        
        # Initialize optimizers
        opt_g, opt_d = self.optimizers()
        
        # Sample noise
        z = torch.randn(real_imgs.shape[0], self.latent_dim)
        z = z.type_as(real_imgs)

        # Train generator
        if self.current_epoch >= self.epoch_start_g_train:
            fake_imgs = self(z)
            y_hat = self.discriminator(fake_imgs)
            
            # Log generated imgs
            sample_imgs = fake_imgs[:9]
            
            y = torch.ones(real_imgs.size(0), 1)
            y = y.type_as(real_imgs)
            
            self.toggle_optimizer(opt_g)
            g_loss = self.adversarial_loss(y_hat, y)
            self.log("g_loss", g_loss, on_epoch=True)
            self.manual_backward(g_loss)
            opt_g.step()
            opt_g.zero_grad()
            self.untoggle_optimizer(opt_g)
        

        # Train discriminator    
        self.toggle_optimizer(opt_d)
        
        for _ in range(self.discriminator_train_freq):
            # Performance of labelling real
            y_hat_real = self.discriminator(real_imgs)
            
            y_real = torch.ones(real_imgs.size(0), 1)
            y_real = y_real.type_as(real_imgs)
            
            real_loss = self.adversarial_loss(y_hat_real, y_real)
            
            # Performance of labelling fake
            y_hat_fake = self.discriminator(self(z).detach())
            
            y_fake = torch.zeros(real_imgs.size(0), 1)
            y_fake = y_fake.type_as(real_imgs)
            
            fake_loss = self.adversarial_loss(y_hat_fake, y_fake)
            
            # Total loss
            d_loss = (real_loss + fake_loss)/2
            self.log("d_loss", d_loss, on_epoch=True)
            
            self.manual_backward(d_loss)
            opt_d.step()
            opt_d.zero_grad()
        self.untoggle_optimizer(opt_d)
        
        # Log losses
        self.epoch_g_losses.append(g_loss.cpu().detach().numpy())
        self.epoch_d_losses.append(d_loss.cpu().detach().numpy())
            
    def configure_optimizers(self):
        lr = self.lr
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr)
        return [opt_g, opt_d], [] # Empty list for scheduler
        
    def on_train_epoch_end(self):
        # Generate samples
        z = self.validation_z.type_as(self.generator.linear[0].weight)
        gen_sample_imgs = self(z)[:9]
        
        # Plot
        self.plot_imgs(gen_sample_imgs, self.real_sample_imgs)
        
        # Log losses
        self.epoch_g_losses, self.epoch_d_losses = self.log_losses(
            self.epoch_g_losses, self.epoch_d_losses)

        # Log sampled images
        grid = torchvision.utils.make_grid(gen_sample_imgs)
        wandb.log({"validation_generated_images": wandb.Image(grid, caption=f"generated_images_{self.current_epoch}")})

class CWGAN(pl.LightningModule, GAN_utils):
    def __init__(self, **training_params):
        super().__init__()
        self.automatic_optimization = False
        
        # Initialize params
        self.latent_dim = training_params['latent_dim']
        self.lr = training_params['lr']
        self.betas = training_params['betas']
        self.gp_lambda = training_params['gp_lambda']
        self.epoch_start_g_train = training_params['epoch_start_g_train']
        self.discriminator_train_freq = training_params['discriminator_train_freq']
        self.log_folder = training_params['model_name']     
        
        gen_version = training_params['generator_version']
        dis_version = training_params['discriminator_version']
        self.root_path = training_params['root_path']
        
        self.epoch_d_losses = []
        self.epoch_g_losses = []
        
        # Initialize models
        self.generator = models.models[f'gen_v{gen_version}'](**training_params)
        self.discriminator = models.models[f'dis_v{dis_version}'](**training_params)

        # Random noise
        self.validation_z = torch.randn(9, training_params['latent_dim'])
        
    def forward(self, z):
        return self.generator(z)

    def gradient_penalty(self, discriminator, real, fake):
        batch_size, c, h, w = real.size()
        epsilon = torch.rand((batch_size, 1, 1, 1)).repeat(1, c, h, w)
        interpolated_imgs = real*epsilon + fake*(1-epsilon)
        
        # calculate discriminator scores
        mixed_scores = discriminator(interpolated_imgs)
        
        gradient = torch.autograd.grad(
            inputs = interpolated_imgs,
            outputs = mixed_scores,
            grad_outputs = torch.ones_like(mixed_scores),
            create_graph = True,
            retain_graph = True
        )[0]
        
        gradient = gradient.view(gradient.shape[0], -1)
        gradient_norm = gradient.norm(2, dim=1)
        gradient_penalty = torch.mean((gradient_norm-1) ** 2)
        
        return gradient_penalty
    
    def training_step(self, batch, batch_idx):
        # load real imgs
        if len(batch) == 2: # if label exists eg. MNIST dataset
            real_imgs, _ = batch
        else:
            real_imgs = batch
            
        real_imgs.requires_grad_()
        
        # Log real imgs
        if batch_idx==0:
            sample_imgs = real_imgs[:9]
            grid = torchvision.utils.make_grid(sample_imgs)
            wandb.log({"real_images": wandb.Image(grid, caption="real_images")})
            self.real_sample_imgs = sample_imgs
        
        # initialize optimizers
        opt_g, opt_d = self.optimizers()
        
        # generate fakes
        z = torch.randn(real_imgs.shape[0], self.latent_dim)
        z = z.type_as(real_imgs)
        
        # train discriminator    
        self.toggle_optimizer(opt_d)
        
        for _ in range(self.discriminator_train_freq):
            dis_real = self.discriminator(real_imgs)
            dis_fake = self.discriminator(self(z).detach())
            
            # calculate loss
            gp = self.gradient_penalty(self.discriminator, real_imgs, self(z))
            d_loss = (
                -(torch.mean(dis_real) - torch.mean(dis_fake)) + self.gp_lambda*gp
            )
            
            # log
            self.log("d_loss", d_loss, on_epoch=True)
            self.log("gp", gp, on_epoch=True)
            
            # update weights
            opt_d.zero_grad()
            self.manual_backward(d_loss, retain_graph=True)
            opt_d.step()

        self.untoggle_optimizer(opt_d)

        # train generator
        if self.current_epoch >= self.epoch_start_g_train:
            fake_imgs = self(z)
            output = self.discriminator(fake_imgs)
            
            self.toggle_optimizer(opt_g)
            
            # calculate loss
            g_loss = -torch.mean(output)
            
            # log
            self.log("g_loss", g_loss, on_epoch=True)
            
            # update weights
            opt_g.zero_grad()
            self.manual_backward(g_loss)
            opt_g.step()
            
            self.untoggle_optimizer(opt_g)
        
        # Log losses
        self.epoch_g_losses.append(g_loss.cpu().detach().numpy())
        self.epoch_d_losses.append(d_loss.cpu().detach().numpy())
        
    def configure_optimizers(self):
        lr = self.lr
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=self.betas)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=self.betas)
        return [opt_g, opt_d], [] # Empty list for scheduler
        
    def on_train_epoch_end(self):
        # Generate samples
        z = self.validation_z.type_as(self.generator.linear[0].weight)
        gen_sample_imgs = self(z)[:9]
        
        # Plot
        self.plot_imgs(gen_sample_imgs, self.real_sample_imgs)
        
        # Log losses
        self.epoch_g_losses, self.epoch_d_losses = self.log_losses(
            self.epoch_g_losses, self.epoch_d_losses)

        # Log sampled images
        grid = torchvision.utils.make_grid(gen_sample_imgs)
        wandb.log({"validation_generated_images": wandb.Image(grid, caption=f"generated_images_{self.current_epoch}")})

gans = {
    'CGAN': CGAN,
    'CWGAN': CWGAN 
}