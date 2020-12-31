import torch.nn as nn
import torch
import os


class VAE(nn.Module):
    def __init__(self, num_channals, latent_dim):
        super(VAE, self).__init__()
        self.num_channals = num_channals
        self.latent_dim = latent_dim
        # Define the architecture of VAE here
        # TODO START
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1), nn.ReLU())
        self.enc2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1), nn.ReLU())
        self.enc3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1), nn.ReLU())
        self.enc4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1), nn.ReLU())
        self.enc5 = nn.Sequential(
            nn.Linear(in_features=16 * 8 * 8, out_features=2 * latent_dim), nn.ReLU())
        self.enc6 = nn.Sequential(
            nn.Linear(in_features=2 * latent_dim, out_features=2 * latent_dim))

        self.enc_conv = nn.Sequential(
            self.enc1, self.enc2, self.enc3, self.enc4
        )

        self.enc_lin = nn.Sequential(
            self.enc5, self.enc6
        )

        self.dec_lin = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=latent_dim), nn.ReLU(),
            nn.Linear(in_features=latent_dim, out_features=16 * 8 * 8),
        )

        self.dec_conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.Sigmoid()
        )

        # All MLP
        # self.enc = nn.Sequential(
        #     nn.Linear(in_features=16 * 8 * 8, out_features=400), nn.ReLU(),
        #     nn.Linear(in_features=400, out_features=latent_dim * 2), nn.ReLU()
        # )
        #
        # self.dec = nn.Sequential(
        #     nn.Linear(in_features=latent_dim, out_features=400), nn.ReLU(),
        #     nn.Linear(in_features=400, out_features=32 * 32), nn.Sigmoid()
        # )

        # TODO END

    def reparameterize(self, mu, log_var):
        '''
        *   Arguments:
            *   mu (torch.FloatTensor): [batch_size, latent_dim], parameters of the diagnoal Gaussian posterior q(z|x)
            *   log_var (torch.FloatTensor): [batch_size, latent_dim], parameters of the diagnoal Gaussian posterior q(z|x)
        *   Returns:
            *   reparameterized samples (torch.FloatTensor): [batch_size, latent_dim]
        '''
        # TODO START
        std = torch.exp(0.5 * log_var)
        sampled_z = torch.randn_like(std) * std + mu
        return sampled_z
        # TODO END

    def forward(self, x=None, z=None):
        '''
        *   Arguments:
            *   x (torch.FloatTensor): [batch_size, 1, 32, 32]
            *   z (torch.FloatTensor): [batch_size, latent_dim]
        *   Returns:
            *   if `x` is not `None`, return a list:
                *   Reconstruction of `x` (torch.FloatTensor)
                *   mu (torch.FloatTensor): [batch_size, latent_dim], parameters of the diagnoal Gaussian posterior q(z|x)
                *   log_var (torch.FloatTensor): [batch_size, latent_dim], parameters of the diagnoal Gaussian posterior q(z|x)
            *  if `x` is `None`, return samples generated from the given `z` (torch.FloatTensor): [num_samples, 1, 32, 32]
        '''
        if x is not None:
            # TODO START
            # print(x.size())
            x = self.enc_conv(x)
            # print(x.size())
            x = x.reshape((-1, 16 * 8 * 8))
            # print(x.size())
            x = self.enc_lin(x)
            # print(x.size())
            mu = x[:, :self.latent_dim]
            log_var = x[:, self.latent_dim: self.latent_dim * 2]
            x_repara = self.reparameterize(mu, log_var)

            recon_x = self.dec_lin(x_repara).reshape((-1, 16, 8, 8))
            recon_x = self.dec_conv(recon_x)

            # All MLP
            # x = self.enc(x.reshape((x.size()[0], -1)))
            # mu = x[:, :self.latent_dim]
            # log_var = x[:, self.latent_dim: self.latent_dim * 2]
            # x_repara = self.reparameterize(mu, log_var)
            # recon_x = self.dec(x_repara).reshape((-1, 1, 32, 32))

            return recon_x, mu, log_var
            # TODO END
        else:
            assert z is not None
            # TODO START
            gen_x = self.dec_lin(z).reshape((-1, 16, 8, 8))
            gen_x = self.dec_conv(gen_x)

            # All MLP
            # gen_x = self.dec(z).reshape((-1, 1, 32, 32))
            return gen_x
            # TODO END

    def restore(self, ckpt_dir):
        try:
            if os.path.exists(os.path.join(ckpt_dir, 'pytorch_model.bin')):
                path = os.path.join(ckpt_dir, 'pytorch_model.bin')
            else:
                path = os.path.join(ckpt_dir, str(max(int(name) for name in os.listdir(ckpt_dir))), 'pytorch_model.bin')
        except:
            return None
        self.load_state_dict(torch.load(path))
        return os.path.split(path)[0]

    def save(self, ckpt_dir, global_step):
        os.makedirs(os.path.join(ckpt_dir, str(global_step)), exist_ok=True)
        path = os.path.join(ckpt_dir, str(global_step), 'pytorch_model.bin')
        torch.save(self.state_dict(), path)
        return os.path.split(path)[0]
