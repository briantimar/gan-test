import torch

def do_D_step(x, G, D, D_optimizer):
    z = torch.randn(x.size(0), G.noise_dimension)
    #fake data
    disc_loss = (-D(x).log() - (1 - D(G(z))).log()).mean()
    
    D.zero_grad()
    disc_loss.backward()
    D_optimizer.step()
    return disc_loss

def do_G_step(x, G, D, G_optimizer):
    
    z = torch.randn(x.size(0), G.noise_dimension)
    gen_loss = - (D(G(z)).log()).mean()
    G.zero_grad()
    gen_loss.backward()
    G_optimizer.step()
    return gen_loss


def do_training_step(x, G, D, G_optimizer, D_optimizer):
    """ Run a single GAN training step.
        x = a batch of real training data
        z = a batch of noise vectors (same size) to be fed to the generator
        Returns: disc_loss, gen_loss"""

    disc_loss = do_D_step(x, G, D, D_optimizer)
    gen_loss = do_G_step(x, G, D, G_optimizer)
    return disc_loss, gen_loss

