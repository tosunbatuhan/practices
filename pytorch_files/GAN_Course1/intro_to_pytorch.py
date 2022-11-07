# a tensor is an image with three color channels with 64 pixels wide and 64 pixels tall is a 3x64x64 tensor

# accessing the PyTorch framework
import torch

###################################
######## TENSOR PROPERTIES ########
###################################

example_tensor = torch.Tensor(
    [
        [[1,2],[3,4]],
        [[5,6],[7,8]],
        [[9,0],[1,2]]
    ]
)

print(example_tensor)

# Device
print(example_tensor.device)

# Shape
print(example_tensor.shape)
print(example_tensor.size(1))

# To learn the rank and number of elements
print("Rank =", len(example_tensor.shape))
print("Number of elements =", example_tensor.numel())

###################################
######## INDEXING TENSORS ########
###################################

print(example_tensor[1])
print(example_tensor[1,1,0])

# If you would like to get a python scalar value use .item()
print(example_tensor = example_tensor[1,1,0].item())

######################################
######## INITIALIZING TENSORS ########
######################################

# to create a tensor of all ones with the same shape and device as example_tensor
print(torch.ones_like(example_tensor))

# to create a tensor of all zeros with the same shape and device as example_tensor
print(torch.zeros_like(example_tensor))

# to create a tensor with every element sampled form a normal distribution with the same shape as example_tensor
print(torch.randn_like(example_tensor))

# also
print(torch.randn(2,2,device='cpu')) # 2x2 tensor

#################################
######## BASIC FUNCTIONS ########
#################################

example_tensor -= 5
example_tensor *= 2

# to calculate the mean and the standard deviation
print("Mean: ", example_tensor.mean())
print("Stdev: ", example_tensor.std())

# you can also find the mean or std along a particular dimension
# if you want to get the average 2x2 matrix of the 3x2x2 example_tensor

print(example_tensor.mean(dim=0))

###############################################
######## PYTORCH NEURAL NETWORK MODULE ########
###############################################

import torch.nn as nn

# nn.Linear
# to create a linear layer
linear = nn.Linear(10,2) # will take n x 10 matrix and return n x 2 matrix
example_input = torch.randn(3, 10)
example_output = linear(example_input)
print(example_output.shape)

# nn.ReLU
# to perform ReLU activation function => non-linear

relu = nn.ReLU()
relu_output = relu(example_output)
print(relu_output)

# nn.BatchNorm1d
# to perform normalization technique that will rescale a batch of n inputs to have a consistent mean and std between batches
# here 1d is for inputs that are vector

batchnorm = nn.BatchNorm1d(2) # takes an argument of the number of input dimensions of each object in the batch
batchnorm_output = batchnorm(relu_output)
print(batchnorm_output)

# nn.Sequential
# to perform a sequence of operations
mlp_layer = nn.Sequential(
    nn.Linear(5,2),
    nn.BatchNorm1d(2),
    nn.ReLU()
)

test_example = torch.randn(5,5) + 1 
output = mlp_layer(test_example)

##############################
######## OPTIMIZATION ########
##############################

# Optimizers
# to create an optimizer in PyTorch, you'll need to use the torch.optim module
# optim.Adam corresponds to the Adam optimizer
# lr = learning rate 

import torch.optim as optim

# for all nn objects, you can access their parameters as a a list using their parameters() method
adam_opt = optim.Adam(mlp_layer.parameters(), lr=1e-1)

# Training Loop
# A basic training step in PyTorch consists of four basic parts
# 1) set all of the gradients to zero using opt.zero_grad()
# 2) calculate the loss, loss
# 3) calculate the gradients wrt the loss using loss.backwards()
# 4) update the parameters being optimized using opt.step()


train_example = torch.randn(100,5)+1
adam_opt.zero_grad()
# we'll use a simple loss function of mean distance from 1
# torch.abs takes the absolute value of a tensor

cur_loss = torch.abs(1 - mlp_layer(train_example)).mean()
cur_loss.backward()
adam_opt.step()
print(cur_loss)

# requires_grad_()
# to tell PyTorch that it needs to calcuclate the gradient
# wrt a tensor that you created by saying

# with torch.no_grad()
# PyTorch usually calculates the gradietns as it proceeds through a set of operations.
# This can often take up unnecessary computations and memory, especially if you're performing an evaluation.
# However, you can wrap a piece of code with with torch.no_grad() to prevent the gradients from being calculated in a piece of code.

# detach():
# sometimes you want to calculate and use a tensor's value without calculating its gradients
# e.g. if you have two models, A and B, and you want to directly optimize the parameters of A wrt the output of B, without calculating through B
# then you could feed the detached output of B to A

################################
######## NEW NN CLASSES ########
################################

# The __init__ function defines what will happen when the object is created
# forward function defines what runs if you create that objectc model and pass it a tensor x

class ExampleModule(nn.Module):
    def __init__(self, input_dims, output_dims):
        super(ExampleModule, self).__init__()
        self.linear = nn.Linear(input_dims, output_dims)
        self.exponent = nn.Parameter(torch.tensor(1,))

    def forward(self, x):
        x = self.linear()
        x = x**self.exponent
        return x

example_model = ExampleModule(10,2)
list(example_model.parameters())

# you can print out the names of the parameters as well
print(list(example_model.named_parameters()))

