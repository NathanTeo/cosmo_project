"""
Author: Nathan Teo

"""

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import numpy as np

import code_model.models as models

class GAN_utils():
    def __init__(self):
        pass
    
    def plot_imgs(self):
        z = self.validation_z.type_as(self.generator.linear[0].weight)
        sample_imgs = self(z).cpu()
        
        fig = plt.figure()
        for i in range(sample_imgs.size(0)):
            plt.subplot(2, 3, i+1)
            plt.imshow(sample_imgs.detach()[i, 0, :, :], interpolation='none')
            plt.title(
                'disc output: {:.4f}'.format(
                    self.discriminator(sample_imgs)[i].detach().numpy()[0]
                )
                )
            plt.xticks([])
            plt.yticks([])
            plt.axis('off')
            plt.tight_layout()
        fig.suptitle(f'Generated Data, Epoch: {self.current_epoch}')
        plt.tight_layout()
        plt.savefig(f'{self.root_path}\\logs\\{self.log_folder}\\images\\image_epoch{self.current_epoch}.png')
        plt.close('all')
    
    def log_losses(self, epoch_g_losses, epoch_d_losses):
        filename = f'{self.root_path}\\logs\\{self.log_folder}\\losses.npz'
        
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
        self.log_folder = training_params['chkpt_file_name']
        
        gen_version = training_params['generator_version']
        dis_version = training_params['discriminator_version']
        
        self.epoch_d_losses = []
        self.epoch_g_losses = []
        
        # Initialize models
        self.generator = models.models[f'gen_v{gen_version}'](**training_params)
        self.discriminator = models.models[f'dis_v{dis_version}'](**training_params)

        # Random noise
        self.validation_z = torch.randn(6, training_params['latent_dim'])
        
    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)
    
    def training_step(self, batch, batch_inx):
        # load real imgs
        if len(batch) == 2: # if label exists eg. MNIST dataset
            real_imgs, _ = batch
        else:
            real_imgs = batch
        
        # Log real imgs
        sample_imgs = real_imgs[:6]
        grid = torchvision.utils.make_grid(sample_imgs)
        self.logger.experiment.add_image("real_images", grid, 0)
        
        # initialize optimizers
        opt_g, opt_d = self.optimizers()
        
        # sample noise
        z = torch.randn(real_imgs.shape[0], self.latent_dim)
        z = z.type_as(real_imgs)

        # train generator
        if self.current_epoch >= self.epoch_start_g_train:
            fake_imgs = self(z)
            y_hat = self.discriminator(fake_imgs)
            
            # Log generated imgs
            sample_imgs = fake_imgs[:6]
            grid = torchvision.utils.make_grid(sample_imgs)
            self.logger.experiment.add_image(f"generated_images_{self.current_epoch}", grid, 0)
            
            y = torch.ones(real_imgs.size(0), 1)
            y = y.type_as(real_imgs)
            
            self.toggle_optimizer(opt_g)
            g_loss = self.adversarial_loss(y_hat, y)
            self.log("g_loss", g_loss, prog_bar=True)
            self.manual_backward(g_loss)
            opt_g.step()
            opt_g.zero_grad()
            self.untoggle_optimizer(opt_g)
        

        # train discriminator    
        self.toggle_optimizer(opt_d)
        
        for _ in range(self.discriminator_train_freq):
            # performance of labelling real
            y_hat_real = self.discriminator(real_imgs)
            
            y_real = torch.ones(real_imgs.size(0), 1)
            y_real = y_real.type_as(real_imgs)
            
            real_loss = self.adversarial_loss(y_hat_real, y_real)
            
            # performance of labelling fake
            y_hat_fake = self.discriminator(self(z).detach())
            
            y_fake = torch.zeros(real_imgs.size(0), 1)
            y_fake = y_fake.type_as(real_imgs)
            
            fake_loss = self.adversarial_loss(y_hat_fake, y_fake)
            
            # total loss
            d_loss = (real_loss + fake_loss)/2
            self.log("d_loss", d_loss, prog_bar=True)
            
            self.manual_backward(d_loss)
            opt_d.step()
            opt_d.zero_grad()
        self.untoggle_optimizer(opt_d)
        
        # Log losses
        self.epoch_g_losses.append(g_loss.detach().numpy())
        self.epoch_d_losses.append(d_loss.detach().numpy())
            
    def configure_optimizers(self):
        lr = self.lr
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr)
        return [opt_g, opt_d], [] # Empty list for scheduler
        
    def on_train_epoch_end(self):
        self.plot_imgs()
        self.epoch_g_losses, self.epoch_d_losses = self.log_losses(
            self.epoch_g_losses, self.epoch_d_losses)
        
        z = self.validation_z.type_as(self.generator.linear[0].weight)

        # log sampled images
        sample_imgs = self(z)
        grid = torchvision.utils.make_grid(sample_imgs)
        self.logger.experiment.add_image(f"val_generated_images_{self.current_epoch}", grid, self.current_epoch)

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
        self.log_folder = training_params['chkpt_file_name']     
        
        gen_version = training_params['generator_version']
        dis_version = training_params['discriminator_version']
        self.root_path = training_params['root_path']
        
        self.epoch_d_losses = []
        self.epoch_g_losses = []
        
        # Initialize models
        self.generator = models.models[f'gen_v{gen_version}'](**training_params)
        self.discriminator = models.models[f'dis_v{dis_version}'](**training_params)

        # Random noise
        self.validation_z = torch.randn(6, training_params['latent_dim'])
        
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
    
    def training_step(self, batch, batch_inx):
        # load real imgs
        if len(batch) == 2: # if label exists eg. MNIST dataset
            real_imgs, _ = batch
        else:
            real_imgs = batch
            
        real_imgs.requires_grad_()
        
        # Log real imgs
        sample_imgs = real_imgs[:6]
        grid = torchvision.utils.make_grid(sample_imgs)
        self.logger.experiment.add_image("real_images", grid, 0)
        
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
            self.log("d_loss", d_loss, prog_bar=True)
            
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
            self.log("g_loss", g_loss, prog_bar=True)
            
            # update weights
            opt_g.zero_grad()
            self.manual_backward(g_loss)
            opt_g.step()
            
            self.untoggle_optimizer(opt_g)
        
        # Log losses
        self.epoch_g_losses.append(g_loss.detach().numpy())
        self.epoch_d_losses.append(d_loss.detach().numpy())
        
    def configure_optimizers(self):
        lr = self.lr
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=self.betas)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=self.betas)
        return [opt_g, opt_d], [] # Empty list for scheduler
        
    def on_train_epoch_end(self):
        self.plot_imgs()
        self.epoch_g_losses, self.epoch_d_losses = self.log_losses(
            self.epoch_g_losses, self.epoch_d_losses)
        
        z = self.validation_z.type_as(self.generator.linear[0].weight)

        # log sampled images
        sample_imgs = self(z)
        grid = torchvision.utils.make_grid(sample_imgs)
        self.logger.experiment.add_image(f"val_generated_images_{self.current_epoch}", grid, self.current_epoch)

gans = {
    'CGAN': CGAN,
    'CWGAN': CWGAN 
}