import torch
import torch.nn as nn
import torch.nn.functional as F
import os

import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torchvision.utils import save_image
import numpy as np
from tqdm import tqdm


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


class Trainer(object):
    def __init__(self, device, netG, netD, optimG, optimD, dataset, ckpt_dir, tb_writer):
        self._device = device
        self._netG = netG
        self._netD = netD
        self._optimG = optimG
        self._optimD = optimD
        self._dataset = dataset
        self._ckpt_dir = ckpt_dir
        self._tb_writer = tb_writer
        os.makedirs(ckpt_dir, exist_ok=True)
        self._netG.restore(ckpt_dir)
        self._netD.restore(ckpt_dir)

    def train_step(self, real_imgs, fake_imgs, BCE_criterion):
        """DO NOT FORGET TO ZERO_GRAD netD and netG
        *   Returns:
            *   loss of netD (scalar)
            *   loss of netG (scalar)
            *   average D(real_imgs) before updating netD
            *   average D(fake_imgs) before updating netD
            *   average D(fake_imgs) after updating netD
        """
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # clear gradients
        self._netD.zero_grad()

        # compute the gradients of binary_cross_entropy(netD(real_imgs), 1) w.r.t. netD
        # record average D(real_imgs)
        # TODO START
        output = self._netD(real_imgs)
        one_ = torch.ones_like(output, device=self._device, dtype=torch.float)
        loss_D_real = BCE_criterion(output, one_)
        D_x = output.mean().item()
        loss_D_real.backward(retain_graph=True)
        # TODO END

        # ** accumulate ** the gradients of binary_cross_entropy(netD(fake_imgs), 0) w.r.t. netD
        # record average D(fake_imgs)
        # TODO START
        output_f = self._netD(fake_imgs)
        zero_ = torch.zeros_like(output_f, device=self._device, dtype=torch.float)
        loss_D_fake = BCE_criterion(output_f, zero_)
        D_G_z1 = output_f.mean().item()
        loss_D_fake.backward(retain_graph=True)
        # TODO END

        # update netD
        self._optimD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        # cl3ear gradients
        self._netG.zero_grad()

        # compute the gradients of binary_cross_entropy(netD(fake_imgs), 1) w.r.t. netG
        # record average D(fake_imgs)
        # TODO START
        output_fake = self._netD(fake_imgs)
        one_ = torch.ones_like(output_fake, device=self._device, dtype=torch.float)
        loss_G = BCE_criterion(output_fake, one_)
        D_G_z2 = output_fake.mean().item()
        loss_G.backward(retain_graph=True)
        # TODO END

        # update netG
        self._optimG.step()

        # return what are specified in the docstring
        return loss_D_real, loss_D_fake, loss_G, D_x, D_G_z1, D_G_z2


    def train(self, num_training_updates, logging_steps, saving_steps):
        fixed_noise = torch.randn(32, self._netG.latent_dim, 1, 1, device=self._device)
        criterion = nn.BCELoss()
        iterator = iter(cycle(self._dataset.training_loader))
        for i in tqdm(range(num_training_updates), desc='Training'):
            inp, _ = next(iterator)
            self._netD.train()
            self._netG.train()
            real_imgs = inp.to(self._device)
            fake_imgs = self._netG(torch.randn(real_imgs.size(0), self._netG.latent_dim, 1, 1))
            errDr, errDf, errG, D_x, D_G_z1, D_G_z2 = self.train_step(real_imgs, fake_imgs, criterion)
            errD = errDf + errDr
            if (i + 1) % logging_steps == 0:
                self._tb_writer.add_scalar("discriminator_loss", errD, global_step=i)
                self._tb_writer.add_scalar("generator_loss", errG, global_step=i)
                self._tb_writer.add_scalar("D(x)", D_x, global_step=i)
                self._tb_writer.add_scalar("D(G(z1))", D_G_z1, global_step=i)
                self._tb_writer.add_scalar("D(G(z2))", D_G_z2, global_step=i)
                self._tb_writer.add_scalar("discriminator_loss_real", errDr, global_step=i)
                self._tb_writer.add_scalar("discriminator_loss_fake", errDf, global_step=i)
            if (i + 1) % saving_steps == 0:
                dirname = self._netD.save(self._ckpt_dir, i)
                dirname = self._netG.save(self._ckpt_dir, i)
                self._netG.eval()
                imgs = make_grid(self._netG(fixed_noise)) * 0.5 + 0.5
                self._tb_writer.add_image('samples', imgs, global_step=i)
                save_image(imgs, os.path.join(dirname, "samples.png"))
        # Interpolation
        N = 10
        self._netG.eval()
        noise1 = torch.randn(32, self._netG.latent_dim, 1, 1, device=self._device)
        noise2 = torch.randn(32, self._netG.latent_dim, 1, 1, device=self._device)
        for k in range(0, N):
            noise = noise1 + k * 1.0 / N * (noise2 - noise1)
            imgs = make_grid(self._netG(noise)) * 0.5 + 0.5
            self._tb_writer.add_image('interpolation', imgs, global_step=k)
