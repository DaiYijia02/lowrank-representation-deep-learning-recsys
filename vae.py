import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from typing import List, Callable, Union, Any, TypeVar, Tuple

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def to_onehot(labels, num_classes, device):

    labels_onehot = torch.zeros(labels.size()[0], num_classes).to(device)
    labels_onehot.scatter_(1, labels.view(-1, 1), 1)

    return labels_onehot


class VAE(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 latent_dim: int,
                 hidden_dim: int,
                 temperature: float = 0.5,
                 anneal_rate: float = 3e-5,
                 anneal_interval: int = 100,  # every 100 batches
                 alpha: float = 30.,
                 **kwargs) -> None:
        super(VAE, self).__init__()

        # The mutation rate (temp) is starting from start_temp and is decreasing over time with anneal_rate.
        # It’s lowest possible value is min_temp.
        # Thus, initially the algorithm explores mutations with a higer mutation rate (more variables are randomly mutated).
        # As time passes, the algorithm exploits the best solutions recorded so far (less variables are mutated).

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.temp = temperature
        self.min_temp = temperature
        self.anneal_rate = anneal_rate
        self.anneal_interval = anneal_interval
        self.alpha = alpha

        # ENCODER
        self.hidden_1 = torch.nn.Linear(
            in_channels, hidden_dim)
        self.z_mean = torch.nn.Linear(hidden_dim, latent_dim)
        # in the original paper (Kingma & Welling 2015, we use
        # have a z_mean and z_var, but the problem is that
        # the z_var can be negative, which would cause issues
        # in the log later. Hence we assume that latent vector
        # has a z_mean and z_log_var component, and when we need
        # the regular variance or std_dev, we simply use
        # an exponential function
        self.z_log_var = torch.nn.Linear(hidden_dim, latent_dim)

        # DECODER
        self.linear_3 = torch.nn.Linear(
            latent_dim, hidden_dim)
        self.linear_4 = torch.nn.Linear(
            hidden_dim, out_channels)

    # def encode(self, input) -> List:
    #     """
    #     Encodes the input by passing through the encoder network
    #     and returns the latent codes.
    #     :param input: (Tensor) Input tensor to encoder [B x C x H x W]
    #     :return: (Tensor) Latent code [B x D x Q]
    #     """
    #     print(input)
    #     result = self.encoder(input)
    #     result = torch.flatten(result, start_dim=1)

    #     # Split the result into mu and var components
    #     # of the latent Gaussian distribution
    #     z = self.fc_z(result)
    #     z = z.view(-1, self.latent_dim, self.categorical_dim)
    #     return [z]

    def encoder(self, features):
        # ENCODER
        x = self.hidden_1(features)
        x = F.leaky_relu(x, negative_slope=0.0001)
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        encoded = self.reparameterize(z_mean, z_log_var)
        return z_mean, z_log_var, encoded

    # def decode(self, z):
    #     """
    #     Maps the given latent codes
    #     onto the image space.
    #     :param z: (Tensor) [B x D x Q]
    #     :return: (Tensor) [B x C x H x W]
    #     """
    #     result = self.decoder_input(z)
    #     result = result.view(-1, 8, 512)
    #     result = self.decoder(result)
    #     result = self.final_layer(result)
    #     return result

    def decoder(self, encoded):
        # DECODER
        x = self.linear_3(encoded)
        x = F.leaky_relu(x, negative_slope=0.0001)
        x = self.linear_4(x)
        decoded = torch.sigmoid(x)
        return decoded

    # def reparameterize(self, z, eps: float = 1e-7):
    #     """
    #     Gumbel-softmax trick to sample from Categorical Distribution
    #     :param z: (Tensor) Latent Codes [B x D x Q]
    #     :return: (Tensor) [B x D]
    #     """
    #     # Sample from Gumbel
    #     u = torch.rand_like(z)
    #     g = - torch.log(- torch.log(u + eps) + eps)

    #     # Gumbel-Softmax sample
    #     s = F.softmax((z + g) / self.temp, dim=-1)
    #     s = s.view(-1, self.latent_dim * self.categorical_dim)
    #     return s

    def reparameterize(self, z_mu, z_log_var):
        # Sample epsilon from standard normal distribution
        eps = torch.randn(z_mu.size(0), z_mu.size(1)).to(device)
        # note that log(x^2) = 2*log(x); hence divide by 2 to get std_dev
        # i.e., std_dev = exp(log(std_dev^2)/2) = exp(log(var)/2)
        print(z_mu.shape)
        z = z_mu + eps * torch.exp(z_log_var/2.)
        return z

    # def forward(self, input, **kwargs) -> List:
    #     q = self.encode(input)[0]
    #     z = self.reparameterize(q)
    #     return [self.decode(z), input, q]

    def forward(self, features):

        z_mean, z_log_var, encoded = self.encoder(features)
        decoded = self.decoder(encoded)

        return z_mean, z_log_var, encoded, decoded

    def loss_function(self, z_mean, z_log_var, encoded, decoded, real_data, labels,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """

        # # Convert the categorical codes into probabilities
        # q_p = F.softmax(q, dim=-1)

        # Account for the minibatch samples from the dataset
        kld_weight = kwargs['M_N']
        batch_idx = kwargs['batch_idx']

        # Anneal the temperature at regular intervals
        if batch_idx % self.anneal_interval == 0 and self.training:
            self.temp = np.maximum(self.temp * np.exp(- self.anneal_rate * batch_idx),
                                   self.min_temp)

        # recons_loss = F.mse_loss(recons, input, reduction='mean')

        # # KL divergence between gumbel-softmax distribution
        # eps = 1e-7

        # # Entropy of the logits
        # h1 = q_p * torch.log(q_p + eps)

        # # Cross entropy with the categorical distribution
        # h2 = q_p * np.log(1. / self.categorical_dim + eps)
        # kld_loss = torch.mean(torch.sum(h1 - h2, dim=(1, 2)), dim=0)

        # cost = reconstruction loss + Kullback-Leibler divergence
        kld_loss = (0.5 * (z_mean**2 +
                           torch.exp(z_log_var) - z_log_var - 1)).sum()

        recons_loss = F.binary_cross_entropy(decoded, labels, reduction='sum')

        # kld_weight = 1.2
        loss = self.alpha * recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'KLD': -kld_loss}

    def sample(self,
               num_samples: int,
               current_device: int, **kwargs):
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """

        M = num_samples * self.latent_dim
        np_y = np.zeros(M, dtype=np.float32)
        np_y[range(M)] = 1
        np_y = np.reshape(np_y, [M // self.latent_dim,
                          self.latent_dim])
        z = torch.from_numpy(np_y)

        # z = self.sampling_dist.sample((num_samples * self.latent_dim, ))
        z = z.view(num_samples, self.latent_dim).to(current_device)
        samples = self.decoder(z)
        return samples

    def generate(self, x, **kwargs):
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]
