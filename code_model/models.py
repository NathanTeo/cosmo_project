"""
Author: Nathan Teo

This scripts contains the Pytorch Lightning modules used for training. 
"""

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import wandb
from typing import Any, Dict
from copy import deepcopy
import os

import code_model.networks as networks
from code_model.testers.plotting_utils import *

class ganUtils():
    """
    Utilities that are useful in the GAN modules.
    Contains plotting and logging functions that are common.
    """
    def __init__(self):
        # Nothing initialized (yet)
        pass
    
    # Plots a grid of real and generated images with the discriminator scores
    def _plot_imgs(self, gen_sample_imgs, real_sample_imgs):
        # Get discriminator scores for the samples
        real_disc_scores = self.discriminator(real_sample_imgs).cpu().detach().numpy()[:,0]
        gen_disc_scores = self.discriminator(gen_sample_imgs).cpu().detach().numpy()[:,0]
        
        # Reshape and send sample images to cpu 
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
        
        # Save plots
        plt.savefig(f'{self.root_path}/logs/images/image_epoch{self.current_epoch}.png')
        plt.close('all')
    
    def _log_losses(self, epoch_g_losses, epoch_d_losses):
        # Log save file
        filename = f'{self.root_path}/logs/losses.npz'
        
        # Logging
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
    
    def _add_noise(self, imgs, mean=0, std_dev=0.05):
        """Adds gaussian noise to a series of image samples"""
        return imgs + (std_dev)*torch.randn(*imgs.size(), device=self.device) + torch.Tensor([mean]).type_as(imgs)
    
    def _backup(self):
        os.system(f'rsync -a {self.root_path}/checkpoints/ {self.root_path}/backup/checkpoints --delete')
        os.system(f'rsync -a {self.root_path}/logs/ {self.root_path}/backup/logs --delete')
        print('\ncheckpoints and logs backed up')

class GapAwareScheduler():
    """
    Dynamic gap aware learning rate update method.
    """
    def __init__(self, optimizer, V_ideal, k0=(2, 0.1), k1=(0.1, 0.1)):
        self.optimizer = optimizer
        self.V_ideal = V_ideal
        self.k0 = k0
        self.k1 = k1
        
    def lr_update_function(self, V_ideal, V_d, k0, k1):
        """
        Function that returns the factor to update the learning rate by. 
        The variables k0 and k1 represent the parameters as such: 
        k0 = (f_max, x_max)
        k1 = (h_min, X_min)
        """
        if V_d >= V_ideal:
            x = V_d - V_ideal
            if x >= 1:
                return 1
            else:
                return torch.min(torch.Tensor([(k0[0])**(x/k0[1]), k0[0]]))
        elif V_d <= V_ideal:
            x = V_ideal - V_d
            return torch.max(torch.Tensor([(k1[0])**(x/k1[1]), k1[0]]))
    
    def clip(self, lr, limits=(1e-6, 0.001)):
        if lr>limits[1]:
            return limits[1]
        elif lr<limits[0]:
            return limits[0]
        else:
            return lr
    
    def step(self, lr, V_d):
        new_lr = lr * self.lr_update_function(self.V_ideal, V_d, self.k0, self.k1)
        new_lr = self.clip(new_lr)
        self.optimizer.param_groups[0]['lr'] = new_lr
        return self.lr_update_function(self.V_ideal, V_d, self.k0, self.k1)
        
    def state_dict(self) -> Dict[str, Any]:
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.__dict__.update(state_dict)
        

class CGAN(pl.LightningModule, ganUtils):
    """
    Pytorch Lightning module for training a GAN using JS divergence
    """
    def __init__(self, **training_params):
        super().__init__()
        self.automatic_optimization = False
        
        # Initialize params
        self.latent_dim = training_params['network_params']['latent_dim']
        self.lr = training_params['lr']
        self.root_path = training_params['root_path']
        self.epoch_start_g_train = training_params['epoch_start_g_train']
        self.discriminator_train_freq = training_params['discriminator_train_freq']
        self.log_folder = training_params['model_name']
        self.noise = training_params['noise']
        
        gen_version = training_params['generator_version']
        dis_version = training_params['discriminator_version']
        
        self.sched_k0 = training_params['scheduler_params'][0]
        self.sched_k1 = training_params['scheduler_params'][1]
        self.sched_alpha = training_params['scheduler_params'][2]
        self.sched_start_epoch = training_params['scheduler_params'][3]
        
        self.epoch_d_losses = []
        self.epoch_g_losses = []
        
        # Initialize models
        self.generator = networks.network_dict[f'gen_v{gen_version}'](**training_params)
        self.discriminator = networks.network_dict[f'dis_v{dis_version}'](**training_params)

        # Random noise
        self.validation_z = torch.randn(9, self.latent_dim)
        
        # initialize d_loss estimate with ideal V 
        self.d_loss_est = np.log10(4)
    
    # Generate image
    def forward(self, z):
        return self.generator(z)

    # Loss function
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
        sched_d = self.lr_schedulers()
        
        # Sample latent noise
        z = torch.randn(real_imgs.shape[0], self.latent_dim)
        z = z.type_as(real_imgs)

        # Train discriminator    
        self.toggle_optimizer(opt_d)
        
        for _ in range(self.discriminator_train_freq):
            gen_imgs = self(z).detach()
            
            # Add noise
            if self.noise is not None:
                real_imgs = self._add_noise(real_imgs, *self.noise)
                gen_imgs = self._add_noise(gen_imgs, *self.noise)
            
            # Performance of labelling real
            y_hat_real = self.discriminator(real_imgs)
            
            y_real = torch.ones(real_imgs.size(0), 1)
            y_real = y_real.type_as(real_imgs)
            
            real_loss = self.adversarial_loss(y_hat_real, y_real)
            
            # Performance of labelling fake
            y_hat_fake = self.discriminator(gen_imgs)
            
            y_fake = torch.zeros(real_imgs.size(0), 1)
            y_fake = y_fake.type_as(real_imgs)
            
            fake_loss = self.adversarial_loss(y_hat_fake, y_fake)
            
            # Total loss
            d_loss = (real_loss + fake_loss)/2
            self.log("d_loss", d_loss, on_epoch=False)
            
            # Update weights
            self.manual_backward(d_loss)
            opt_d.step()
            opt_d.zero_grad()
            
            # Update learning rate
            if self.current_epoch >= self.sched_start_epoch:
                self.log("d_lr", self.get_lr(opt_d))
                self.d_loss_est = self.sched_alpha*self.d_loss_est + (1-self.sched_alpha)*d_loss.detach()
                sched_d.step(self.get_lr(opt_d), self.d_loss_est)
            
            
        self.untoggle_optimizer(opt_d)

        # Train generator
        if self.current_epoch >= self.epoch_start_g_train:
            gen_imgs = self(z)
            
            # Add noise
            if self.noise is not None:
                gen_imgs = self._add_noise(gen_imgs, *self.noise)
            
            # Loss
            y_hat = self.discriminator(gen_imgs)
            
            y = torch.ones(real_imgs.size(0), 1)
            y = y.type_as(real_imgs)
            
            g_loss = self.adversarial_loss(y_hat, y)
            self.log("g_loss", g_loss, on_epoch=False)
            
            # Update weights
            self.toggle_optimizer(opt_g)
            
            self.manual_backward(g_loss)
            opt_g.step()
            opt_g.zero_grad()
            
            self.untoggle_optimizer(opt_g)
        
        # Log losses
        self.epoch_g_losses.append(g_loss.cpu().detach().numpy())
        self.epoch_d_losses.append(d_loss.cpu().detach().numpy())       
        
    # Optimizer
    def configure_optimizers(self):
        lr = self.lr
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr)
        sched_d = GapAwareScheduler(opt_d, V_ideal=np.log10(4), k0=self.sched_k0, k1=self.sched_k1)
        return [opt_g, opt_d], [sched_d, ]
    
    def get_lr(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']
    
    # Method is run at the end of each training epoch
    def on_train_epoch_end(self):
        # Generate samples
        z = self.validation_z.type_as(self.generator.linear[0].weight)
        gen_sample_imgs = self(z)[:9]
        
        # Plot
        self._plot_imgs(gen_sample_imgs, self.real_sample_imgs)
        
        # Log losses
        self.epoch_g_losses, self.epoch_d_losses = self._log_losses(
            self.epoch_g_losses, self.epoch_d_losses)

        # Log sampled images
        grid = torchvision.utils.make_grid(gen_sample_imgs)
        wandb.log({"validation_generated_images": wandb.Image(grid, caption=f"generated_images_{self.current_epoch}")})

    def on_test_epoch_start(self):
        self.test_output_list = {
            'gen_imgs': []
        }

    def test_step(self, batch, batch_idx):
        # Load real imgs
        if len(batch) == 2: # if label exists eg. MNIST dataset
            real_imgs, _ = batch
        else:
            real_imgs = batch
            
        # Generate imgs
        z = torch.randn(real_imgs.shape[0], self.latent_dim)
        z = z.type_as(real_imgs)
        gen_imgs = self(z).cpu().detach().squeeze().numpy()
        
        self.test_output_list['gen_imgs'].extend(gen_imgs)

    def on_test_epoch_end(self):
        gen_imgs = self.test_output_list['gen_imgs']
        self.outputs =  np.array(gen_imgs) 

class CWGAN(pl.LightningModule, ganUtils):
    """
    Pytorch Lightning module for training a GAN using Wasserstein distance
    """
    def __init__(self, **training_params):
        super().__init__()
        self.automatic_optimization = False
        
        # Initialize params
        self.latent_dim = training_params['network_params']['latent_dim']
        self.lr = training_params['lr']
        self.ttur_ratio = 10
        self.betas = training_params['betas']
        self.loss_method = training_params['loss_method']
        self.gp_lambda = training_params['gp_lambda']
        self.epoch_start_g_train = training_params['epoch_start_g_train']
        self.discriminator_train_freq = training_params['discriminator_train_freq']
        self.noise = training_params['noise']
        
        gen_version = training_params['generator_version']
        dis_version = training_params['discriminator_version']
        self.root_path = training_params['root_path']
        
        self.sched_k0 = training_params['scheduler_params'][0]
        self.sched_k1 = training_params['scheduler_params'][1]
        self.sched_alpha = training_params['scheduler_params'][2]
        self.sched_start_epoch = training_params['scheduler_params'][3]
         
        self.epoch_d_losses = []
        self.epoch_g_losses = []
        
        # Initialize models
        self.generator = networks.network_dict[f'gen_v{gen_version}'](**training_params)
        self.discriminator = networks.network_dict[f'dis_v{dis_version}'](**training_params)

        # Random noise
        self.validation_z = torch.randn(9, self.latent_dim)
        
        # initialize d_loss estimate with ideal V 
        self.d_loss_est = 0 
    
    # Generate Image
    def forward(self, z):
        """Generate image if model is called"""
        return self.generator(z)

    # Calculate gradient penalty
    def gradient_penalty(self, discriminator, real, fake, method='LP'):
        """Calculate gradient penalty"""
        batch_size, c, h, w = real.size()
        
        # Interpolate image
        epsilon = torch.rand((batch_size, 1, 1, 1), device=self.device).repeat(1, c, h, w)
        interpolated_imgs = real*epsilon + fake*(1-epsilon)
        
        # Calculate discriminator scores
        mixed_scores = discriminator(interpolated_imgs)
        
        # Calculate gradient penalty
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

        # Gulrajani, Ishaan, et al. "Improved training of wasserstein gans." arXiv preprint arXiv:1704.00028 (2017).
        if method == 'GP':
            gradient_penalty = torch.mean((gradient_norm - 1) ** 2)

        # Henning Petzka, Asja Fischer, and Denis Lukovnicov. "On the regularization of Wasserstein GANs."
        # arXiv preprint arXiv:1709.08894 (2017).
        elif method == 'LP':
            gradient_penalty = torch.mean((torch.maximum(torch.tensor(0), gradient_norm - 1)) ** 2)
        
        return gradient_penalty
    
    def training_step(self, batch, batch_idx):
        # Progress
        # print(f'Epoch {self.current_epoch}', end='\t')
        
        # Load real imgs
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
        sched_d = self.lr_schedulers()
        
        # # Sample latent noise
        z = torch.randn(real_imgs.shape[0], self.latent_dim)
        z = z.type_as(real_imgs)
        
        # Train discriminator    
        self.toggle_optimizer(opt_d)
        
        for _ in range(self.discriminator_train_freq):
            gen_imgs = self(z).detach()
            
            # Add noise
            if self.noise is not None:
                real_imgs = self._add_noise(real_imgs, *self.noise)
                gen_imgs = self._add_noise(gen_imgs, *self.noise)
            
            dis_real = self.discriminator(real_imgs)
            dis_fake = self.discriminator(gen_imgs)
            
            # Calculate loss
            if self.loss_method=='w-gp':
                gp = self.gradient_penalty(self.discriminator, real_imgs, self(z), method='GP')
                d_loss = (
                    -(torch.mean(dis_real) - torch.mean(dis_fake)) + self.gp_lambda*gp
                )
                self.log("gp", gp, on_epoch=False)
            elif self.loss_method=='w-lp':
                lp = self.gradient_penalty(self.discriminator, real_imgs, self(z), method='LP')
                d_loss = (
                    -(torch.mean(dis_real) - torch.mean(dis_fake)) + self.gp_lambda*lp
                )
                self.log("lp", lp, on_epoch=False)
            elif self.loss_method=='hinge':
               d_loss_real = torch.nn.ReLU()(1. - dis_real).mean()
               d_loss_fake = torch.nn.ReLU()(1. + dis_fake).mean()
               d_loss = d_loss_real + d_loss_fake
            
            # Log
            self.log("d_loss", d_loss, on_epoch=False)

            # Update weights
            opt_d.zero_grad()
            self.manual_backward(d_loss, retain_graph=True)
            opt_d.step()
            
            # Update learning rate
            if self.current_epoch >= self.sched_start_epoch:
                self.log("d_lr", self.get_lr(opt_d))
                self.d_loss_est = self.sched_alpha*self.d_loss_est + (1-self.sched_alpha)*d_loss.detach()
                update = sched_d.step(self.get_lr(opt_d), self.d_loss_est)
                self.log("update", update)

        self.untoggle_optimizer(opt_d)

        # Train generator
        if self.current_epoch >= self.epoch_start_g_train:
            gen_imgs = self(z)
            
            # Add noise
            if self.noise is not None:
                gen_imgs = self._add_noise(gen_imgs, *self.noise)
            
            output = self.discriminator(gen_imgs)
            
            self.toggle_optimizer(opt_g)
            
            # calculate loss
            if self.loss_method=='w-gp' or self.loss_method=='w-lp':
                g_loss = - torch.mean(output)
            elif self.loss_method=='hinge':
                g_loss = - torch.mean(output)
            
            # log
            self.log("g_loss", g_loss, on_epoch=False)
            
            # update weights
            opt_g.zero_grad()
            self.manual_backward(g_loss)
            opt_g.step()
            
            self.untoggle_optimizer(opt_g)
        
        # Log losses
        self.epoch_g_losses.append(g_loss.cpu().detach().numpy())
        self.epoch_d_losses.append(d_loss.cpu().detach().numpy()) 
        
        ##### FOR TESTING #####
        # try:
        #     z = self.validation_z.type_as(self.generator.linear[0].weight)
        # except TypeError: # BigGAN
        #     z = self.validation_z.type_as(self.generator.linear.W_())
        # gen_sample_imgs = self(z)[:9]
        # self._plot_imgs(gen_sample_imgs, self.real_sample_imgs)
        
    def configure_optimizers(self):
        lr = self.lr
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=self.betas)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.ttur_ratio*lr, betas=self.betas) # TTUR
        sched_d = GapAwareScheduler(opt_d, V_ideal=np.log10(4), k0=self.sched_k0, k1=self.sched_k1)
        return [opt_g, opt_d], [sched_d, ]
    
    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step()
        
    def get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']
    
    # Method is run at the end of each training epoch
    def on_train_epoch_end(self):
        # Generate samples
        try:
            z = self.validation_z.type_as(self.generator.linear[0].weight)
        except TypeError: # BigGAN
            z = self.validation_z.type_as(self.generator.linear.W_())

        gen_sample_imgs = self(z)[:9]
        
        # Plot
        self._plot_imgs(gen_sample_imgs, self.real_sample_imgs)
        
        # Log losses
        self.epoch_g_losses, self.epoch_d_losses = self._log_losses(
            self.epoch_g_losses, self.epoch_d_losses)

        # Log sampled images
        grid = torchvision.utils.make_grid(gen_sample_imgs)
        wandb.log({"validation_generated_images": wandb.Image(grid, caption=f"generated_images_{self.current_epoch}")})
        
        # Backup every 20 epochs
        if ((self.current_epoch+1)%20)==0:
            self._backup()
            
        # Progress
        # print('complete')
    
    def on_test_epoch_start(self):
        self.test_output_list = {
            'gen_imgs': []
        }

    def test_step(self, batch, batch_idx):
        # Load real images
        if len(batch) == 2: # if label exists eg. MNIST dataset
            real_imgs, _ = batch
        else:
            real_imgs = batch
            
        # Generate images
        z = torch.randn(real_imgs.shape[0], self.latent_dim)
        z = z.type_as(real_imgs)
        gen_imgs = self(z).cpu().detach().squeeze().numpy()
        
        self.test_output_list['gen_imgs'].extend(gen_imgs)

    # Track generated output image samples
    def on_test_epoch_end(self):
        gen_imgs = self.test_output_list['gen_imgs']
        self.outputs =  np.array(gen_imgs)
    
    # Get discriminator scores for an input image samples
    def score_samples(self, samples, batch_size=128, progress_bar=False):
        batched_samples = torch.split(torch.unsqueeze(torch.tensor(samples, device=self.device, dtype=torch.float), 1), batch_size)
        scores = np.array([])
        
        for batch in tqdm(batched_samples, desc='scoring', disable=not progress_bar):
            scores = np.append(scores, self.discriminator(batch).cpu().detach().numpy())
        
        return scores

    # Returns the number of learnable parameters in the model
    def count_learnable_params(self):
        gen_params_count = sum(p.numel() for p in self.generator.parameters() if p.requires_grad)
        dis_params_count = sum(p.numel() for p in self.discriminator.parameters() if p.requires_grad)
        return gen_params_count, dis_params_count

class EMA():
    """
    Exponential moving average for diffusion model
    Weights in this network are less suseptible to large changes 
    """
    def __init__(self, beta):
        self.beta = beta
    
    def update_network_average(self, ema_network, network):
        # Update all weights
        for current_param, ema_param in zip(network.parameters(), ema_network.parameters()):
            old_weight, new_weight = ema_param.data, current_param.data
            ema_param.data = self.update_average(old_weight, new_weight)
            
    def update_average(self, old, new):
        # Moving average of weight
        return old*self.beta + (1-self.beta)*new
    
    def step_ema(self, ema_network, network, epoch, epoch_start_ema=10):
        # Step after specified epoch 
        if epoch < epoch_start_ema:
            self.reset_parameters(ema_network, network)
            return
        self.update_network_average(ema_network, network)
        
    def reset_parameters(self, ema_network, network):
        # Reload ema to match network
        ema_network.load_state_dict(network.state_dict())
        
        
class Diffusion(pl.LightningModule):
    def __init__(self, scaling_factor=1, **training_params):
        super().__init__()
        self.scaling_factor=scaling_factor
        
        self.unet_version = training_params['unet_version']
        self.training_params = training_params
        self.root_path = training_params['root_path']
        
        self.img_size = training_params['image_size']
        self.image_channels = training_params['network_params']['image_channels']
        self.noise_steps = training_params['network_params']['noise_steps']
        
        self.lr = training_params['lr']
        self.scheduler_params = training_params['scheduler_params']
        self.loss_fn = torch.nn.MSELoss()

        self.network = networks.network_dict[f'unet_v{self.unet_version}'](scaling_factor=scaling_factor, **self.training_params)
        self.ema = EMA(beta=0.995)
        self.ema_network = deepcopy(self.network).eval().requires_grad_(False)
        
        self.betas = self.cosine_schedule()
        self.alphas = 1. - self.betas
        self.alpha_hats = torch.cumprod(self.alphas, dim=0)
    
        self.epoch_losses = []
    
    def setup(self, stage):
        if stage=='fit':
            self.test_dataloader
        
    def cosine_schedule(self, s=0.008):
        """Prepares cosine scheduler for adding noise"""
        def f(t):
            return torch.cos(((t/self.noise_steps) + s) / (1 + s) * 0.5 * torch.pi) ** 2
        x = torch.linspace(0, self.noise_steps, self.noise_steps + 1, device=self.device)
        alpha_cumprod = f(x) / f(torch.tensor([0], device=self.device))
        betas = 1 - alpha_cumprod[1:] / alpha_cumprod[:-1]
        betas = torch.clip(betas, 0.0001, 0.9999)
        return betas

    def add_noise(self, x, t):
        """Adds gaussian noise to images"""
        sqrt_alpha_hats = torch.sqrt(self.alpha_hats[t])[:,None,None,None]
        sqrt_one_minus_alpha_hats = torch.sqrt(1. - self.alpha_hats[t])[:,None,None,None]
        noise = torch.randn_like(x)
        
        return sqrt_alpha_hats*x + sqrt_one_minus_alpha_hats*noise, noise

    def sample_timesteps(self, n):
        """Return randomly sampled timesteps"""
        return torch.randint(low=1, high=self.noise_steps, size=(n,), device=self.device)
    
    def sample(self, network, n, scale=False):
        """Return n sampled noised image at all timesteps"""
        network.eval()
        with torch.no_grad():
            # Random noise
            x = torch.randn((n, self.image_channels, self.img_size, self.img_size), device=self.device)
            
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                # Tensor of the current t of length n (for each of the samples)
                t = (torch.ones(n, device=self.device)*i).long()
                # Predict noise for current t
                predicted_noise = network(x, t)
                # Get new noise to add
                alpha = self.alphas[t][:,None,None,None]
                alpha_hat = self.alpha_hats[t][:,None,None,None]
                beta = self.betas[t][:,None,None,None]
                if i>1:
                    noise = torch.randn_like(x)
                else: # Don't noise at the last step
                    noise = torch.zeros_like(x)
                # Subtract predicted noise from image and renoise to next t
                x = 1/torch.sqrt(alpha) * (x - ((1-alpha) / (torch.sqrt(1-alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        network.train()
        
        # Restrict output to within -1 and 1
        x = x.clamp(-1, 1)
        
        # Rescale for testing such that blob amplitude consistent across all datasets
        if scale:
            x /= self.scaling_factor
        
        return x
    
    def on_train_epoch_start(self):
        # Move to device
        self.betas = self.betas.to(self.device)
        self.alphas = self.alphas.to(self.device)
        self.alpha_hats = self.alpha_hats.to(self.device)
        
        self.ema_network.device = self.device
        
    def training_step(self, batch, batch_idx):
        # Progress
        # print(f'Epoch {self.current_epoch}', end='\t')
        
        # Load real imgs
        if len(batch)==2: # if label exists eg. MNIST dataset
            real_imgs, _ = batch
        else:
            real_imgs = batch
 
        # Log real imgs
        if batch_idx==0:
            sample_imgs = real_imgs[:9]
            grid = torchvision.utils.make_grid(sample_imgs)
            wandb.log({"real_images": wandb.Image(grid, caption="real_images")})
            self.real_sample_imgs = sample_imgs       
           
        real_imgs.requires_grad_()
        
        # Generate noised images
        t = self.sample_timesteps(real_imgs.shape[0])
        x_t, noise = self.add_noise(real_imgs, t)
        
        # Predict noise
        predicted_noise = self.network(x_t, t)
        
        # Calculate loss
        loss = self.loss_fn(noise, predicted_noise)
        
        # Log loss
        self.epoch_losses.append(loss.cpu().detach().numpy())
        self.log("loss", loss, on_epoch=False)
        
        return loss
        
    def on_train_batch_end(self, outputs, batch, batch_idx):
        # Step exponential moving average
        self.ema.step_ema(self.ema_network, self.network, self.current_epoch)
        
    def configure_optimizers(self):
        # Optimizer and schedulers
        optimizer = torch.optim.AdamW(self.network.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, *self.scheduler_params)
        return [optimizer], [scheduler]
    
    def get_lr(self, optimizer):
        # Returns current learning rate
        for param_group in optimizer.param_groups:
            return param_group['lr']
    
    # Method is run at the end of each training epoch
    def on_train_epoch_end(self):
        # Generate samples
        ema_gen_sample_imgs = self.sample(self.ema_network, n=9)       

        # Plot
        self._plot_imgs(self.real_sample_imgs, ema_gen_sample_imgs)
        
        # Log losses
        self.epoch_losses = self._log_losses(self.epoch_losses)

        # Log sampled images
        grid = torchvision.utils.make_grid(ema_gen_sample_imgs)
        wandb.log({"validation_generated_images": wandb.Image(grid, caption=f"ema_generated_images_{self.current_epoch}")})

        # Backup every 20 epochs
        if ((self.current_epoch+1)%20)==0:
            self._backup()
        
        # Progress
        # print('complete')
    
    def on_test_epoch_start(self):
        self.test_output_list = {
            'gen_imgs': []
        }

        self.betas = self.betas.to(self.device)
        self.alphas = self.alphas.to(self.device)
        self.alpha_hats = self.alpha_hats.to(self.device)
        
        self.network.device = self.device
        self.ema_network.device = self.device
        
    def test_step(self, batch, batch_idx):
        # Load real images
        if len(batch) == 2: # if label exists eg. MNIST dataset
            real_imgs, _ = batch
        else:
            real_imgs = batch
        
        # Generate images
        gen_imgs = self.sample(self.ema_network, n=real_imgs.shape[0], scale=True).cpu().detach().squeeze().numpy()
        
        self.test_output_list['gen_imgs'].extend(gen_imgs)

    # Track generated output image samples
    def on_test_epoch_end(self):
        gen_imgs = self.test_output_list['gen_imgs']
        self.outputs =  np.array(gen_imgs)

    def _plot_imgs(self, real_imgs, gen_imgs):        
        # Reshape and send sample images to cpu 
        real_imgs = real_imgs.cpu().detach()[:,0,:,:]
        gen_imgs = gen_imgs.cpu().detach()[:,0,:,:]
        
        # Plotting grid of images
        fig = plt.figure(figsize=(8,5))
        subfig = fig.subfigures(1, 2, wspace=0.2)
        
        plot_img_grid(
            subfig[0], real_imgs, 3, 
            title='Real Imgs', wspace=.1, hspace=.1
            )
        plot_img_grid(
            subfig[1], gen_imgs, 3,
            title='Generated Imgs', wspace=.1, hspace=.1
            )
        
        fig.suptitle(f'Epoch {self.current_epoch}')
        plt.tight_layout()
        
        # Save plots
        plt.savefig(f'{self.root_path}/logs/images/image_epoch{self.current_epoch}.png')
        plt.close('all')
    
    def _log_losses(self, losses_epoch):
        # Log save file
        filename = f'{self.root_path}/logs/losses.npz'
        
        # Logging
        if self.current_epoch == 0:
            epochs = [0] # Current epoch is 0
            losses = [np.mean(losses_epoch)]
            np.savez(filename, epochs=epochs, losses=losses)
        else:
            f = np.load(filename, allow_pickle=True)
            epochs = np.append(f['epochs'], self.current_epoch)
            losses = np.append(f['losses'], np.mean(losses_epoch))
            np.savez(filename, epochs=epochs, losses=losses)
        
        return [] # For resetting epoch_losses
    
    def _backup(self):
        os.system(f'rsync -a {self.root_path}/checkpoints/ {self.root_path}/backup/checkpoints --delete')
        os.system(f'rsync -a {self.root_path}/logs/ {self.root_path}/backup/logs --delete')
        print('\ncheckpoints and logs backed up')
    
    def count_learnable_params(self):
        params_count  = sum(p.numel() for p in self.network.parameters() if p.requires_grad)
        return params_count 


    
model_dict = {
    'CGAN': CGAN,
    'CWGAN': CWGAN,
    'Diffusion': Diffusion
}