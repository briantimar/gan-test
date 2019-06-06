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
        self.layers += [nn.Linear(self.intermediate_size, self.output_dimension), 
                        nn.LeakyReLU(negative_slope=.5)
                        ]
        for i in range(self.num_hidden_layers+1):
            self.add_module("linear%d"%i, self.layers[2*i])
            self.add_module("activation%d"%i, self.layers[2*i+1])
        
    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x


class Discriminator(nn.Module):
    """ Discrimator for the GAN"""
    
    def __init__(self, input_dimension):
        super().__init__()
        self.input_dimension = input_dimension
        self.intermediate_size = 10
        self.num_hidden_layers = 2
        self.activation = nn.ReLU
        self.layers = [nn.Linear(self.input_dimension, self.intermediate_size ), 
                        self.activation()]
        for __ in range(self.num_hidden_layers-1):
            self.layers += [nn.Linear(self.intermediate_size, self.intermediate_size), self.activation()]
        self.layers += [nn.Linear(self.intermediate_size, 1), nn.Sigmoid()]
        for i in range(self.num_hidden_layers+1):
            self.add_module("linear%d"%i, self.layers[2*i])
            self.add_module("activation%d"%i, self.layers[2*i+1])
        
    def forward(self,x):
        for l in self.layers:
            x = l(x) 
        return x


def do_training_step(x, z, G, D, G_optimizer, D_optimizer):
    """ Run a single GAN training step."""

    #fake data
    xg = G(z)
    disc_loss = -D(x).log() - (1 - D(xg)).log()
    
    D.zero_grad()
    disc_loss.backward()
    D_optimizer.step()

    gen_loss = - D(xg).log()
    G.zero_grad()
    gen_loss.backward()
    G_optimizer.step()



