################################
######## YOUR FIRST GAN ########
################################

# you will build and train a GAN that can generate hand-written images of digits (0-9)

# Learning Objectives
# 1) build the generator and discriminator components of a GAN from scratch
# 2) create generator and discriminator loss functions
# 3) train your GAN and visualize the generated images

#########################
######## IMPORTS ########
#########################

import torch
from torch import nn
# from tqdm.auto import tqdm # we dont have it 
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
torch.manual_seed(0) # for testing purposes

# visualizer function to investigate the images our GAN creates

def show_tensor_images(image_tensor, num_images=25, size=(1,28,28)):
    image_unflat = image_tensor.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1,2,0).squeeze())
    plt.show()

###########################
######## GENERATOR ########
###########################

# each block should include a linear transformation to map to another shape,
# a batch normalization for stabilization, and finally a non-linear activation
# so the output can be transformed in complex ways


def get_generator_block(input_dim, output_dim):

    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.BatchNorm1d(output_dim),
        nn.ReLU(inplace=True)
    )

# now you can build the generator class
# it will take 3 values
# - the noise vector dimension
# - the image dimension 
# - the initial hidden dimension

# using these values the generator will build a nn with 5 layers/blocks
# beginning with the noise vector, the generator will apply non-linear transformations
# via the block function until the tensor is mapped to the size of the image to be outputted
# so you will need to fill in the code for final layer since its different than the others
# the final layer does not need a normalization or activation function
# , but does need to be scaled with a sigmoid


class Generator(nn.Module):
    def __init__(self, z_dim=10, im_dim=784, hidden_dim=128):
        super(Generator, self).__init__()

        self.gen = nn.Sequential(
            get_generator_block(z_dim,hidden_dim),
            get_generator_block(hidden_dim, hidden_dim*2),
            get_generator_block(hidden_dim*2, hidden_dim*4),
            get_generator_block(hidden_dim*4, hidden_dim*8),
            nn.Linear(hidden_dim*8, im_dim),
            nn.Sigmoid()
        )

    def forward(self, noise):
        return self.gen(noise)
    
    # needed for grading
    def get_gen(self):
        return self.gen # returns the sequential model

#######################
######## NOISE ########
#######################

# to be able t ouse your generator, you will need to be able to create noise vectors
# the noise vector z has the important role of making sure the images generated from the same class dont look the same

def get_noise(n_samples, z_dim, device="cpu"):
    return torch.randn(n_samples, z_dim).to(device)

###############################
######## DISCRIMINATOR ########
###############################

# use leaky relu to prevent the "dying relu" problem => where the parameters stop changing due to consistently negative values passed to ReLU
# , which results in a zero gradient 


def get_discriminator_block(input_dim, output_dim):

    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.LeakyReLU(negative_slope=0.2)
    )


# the discriminator class holds 2 values:
# - the image dimension
# - the hidden dimension

# the discriminator will build a neural network with 4 layers 
# it wll start with the image tensor and transform it until it returns a single number
# note that you dont need a sigmoid after the output layer since it is included in the loss function (BCE Loss)

class Discriminator(nn.Module):
    def __init__(self, im_dim=784, hidden_dim=128):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            get_discriminator_block(im_dim, hidden_dim*4),
            get_discriminator_block(hidden_dim*4, hidden_dim*2),
            get_discriminator_block(hidden_dim*2, hidden_dim),
            nn.Linear(hidden_dim,1)
        )

    def forward(self, image):
        return self.disc(image)

    def get_disc(self):
        return self.disc


##########################
######## TRAINING ########
##########################

# set your parameters 
criterion =nn.BCEWithLogitsLoss()
n_epochs = 200
z_dim = 64
display_step = 500
batch_size = 128
lr = 0.00001
# Load MNIST dataset as tensors
dataloader = DataLoader(
    MNIST('.', download=False,transform=transforms.ToTensor()),
    batch_size=batch_size,
    shuffle=True
)
# device
device = 'cuda'
# initialize your generator, discriminator, and optimizers

gen = Generator(z_dim).to(device)
gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)
disc = Discriminator().to(device)
disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)

# Since the generator is needed when calculating the discriminator's loss, you will need to call .detach() on the generator result
# to ensure that only the discriminator is updated!

# to get the discriminator loss

def get_disc_loss(gen, disc, criterion, real, num_images, z_dim, device):

    # gen : the generator model, which returns an image given z-dimensional noise
    # disc : the discriminator model, which returns a single dimensional prediction of real/fake
    # real : a batch of real images
    # num_images : the number of images the generator should produce
    # z_dim : dimension of the noise vector

    # disc_loss : a torch scalar loss value for the current batch

    noise_vectors = get_noise(num_images, z_dim, device=device)
    fake_images = gen(noise_vectors)
    fake_images = fake_images.detach()
    loss_fake = criterion(disc(fake_images), torch.zeros(num_images,1).to(device))
    loss_real = criterion(disc(real), torch.ones(num_images,1).to(device))
    disc_loss = (loss_fake+loss_real)/2
    
    return disc_loss

# to get the generator loss

def get_gen_loss(gen, disc, criterion, num_images, z_dim, device):

    noise_vectors = get_noise(num_images, z_dim, device=device)
    fake_images = gen(noise_vectors)
    disc_preds = disc(fake_images)
    gen_loss  = criterion(disc_preds, torch.ones_like(disc_preds))
    return gen_loss

###############################
######## OPTIONAL PART ########
###############################


cur_step = 0
mean_generator_loss = 0
mean_discriminator_loss = 0
gen_loss = False
error = False

for epoch in range(n_epochs):

    # DataLoader returns the batches
    for real, _ in tqdm(dataloader):
        cur_batch_size = len(real)

        # Flatten the batch of real images from the dataset
        real = real.view(cur_batch_size, -1).to(device)

        # Update discriminator
        # zero out the gradients before backprop
        disc_opt.zero_grad()

        # calculate discriminator loss
        disc_loss = get_disc_loss(gen, disc, criterion, real, cur_batch_size, z_dim, device)

        # update gradients
        disc_loss.backward(retain_graph=True)

        # update optimizer
        disc_opt.step()

        # Update generator
        gen_opt.zero_grad()
        gen_loss = get_gen_loss(gen, disc, criterion, cur_batch_size, z_dim, device)
        gen_loss.backward()
        gen_opt.step()

        # Keep track of the average discriminator loss
        mean_discriminator_loss += disc_loss.item() / display_step

        # Keep track of the average generator loss
        mean_generator_loss += gen_loss.item()/display_step

        # Visualization code
        if cur_step % display_step == 0 and cur_step>0:
            print(f"Step {cur_step}: Generator loss: {mean_generator_loss}, discriminator loss: {mean_discriminator_loss}")
            fake_noise = get_noise(cur_batch_size, z_dim, device=device)
            fake = gen(fake_noise)
            show_tensor_images(fake)
            show_tensor_images(real)
            mean_generator_loss = 0
            mean_discriminator_loss = 0
        cur_step += 1