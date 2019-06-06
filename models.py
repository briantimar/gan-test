import torch
import numpy as np
import torch.nn as nn

class Generator(nn.Module):
    """ A GAN generator. 
        Takes noise vectors as input, returns "image" vectors. """

    def __init__(self, noise_dimension, output_dimension):
        super().__init__()
        self.noise_dimension = noise_dimension
        self.output_dimension = output_dimension

        self.intermediate_size = 10
        self.num_hidden_layers = 2
        self.activation = nn.ReLU
        self.layers = [ nn.Linear(self.noise_dimension, self.intermediate_size), 
                        self.activation()]
        for __ in range(self.num_hidden_layers-1):
            self.layers += [nn.Linear(self.intermediate_size, self.intermediate_size), self.activation()]
        self.layers += [nn.Linear(self.intermediate_size, self.output_dimension)
                        ]
        
    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x


class Discriminator(nn.Module):
    """ Discrimator for the GAN"""
    pass