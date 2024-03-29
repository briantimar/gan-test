import torch
import numpy as np
import torch.nn as nn

class Generator(nn.Module):
    """ A GAN generator. 
        Takes noise vectors as input, returns "image" vectors. """

    def __init__(self, noise_dimension, output_dimension,num_hidden_layers=2):
        super().__init__()
        self.noise_dimension = noise_dimension
        self.output_dimension = output_dimension

        self.intermediate_size = 10
        self.num_hidden_layers = num_hidden_layers
        def get_activation():
            return nn.LeakyReLU(negative_slope=.1)

        self.layers = [ nn.Linear(self.noise_dimension, self.intermediate_size), 
                        get_activation()]
        for __ in range(self.num_hidden_layers-1):
            self.layers += [nn.Linear(self.intermediate_size, self.intermediate_size), get_activation()]
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
    
    def __init__(self, input_dimension, num_hidden_layers=2):
        super().__init__()
        self.input_dimension = input_dimension
        self.intermediate_size = 10
        self.num_hidden_layers = num_hidden_layers
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




