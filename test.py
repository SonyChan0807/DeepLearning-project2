import os
import sys

cwd = os.getcwd()
sys.path.append(cwd)
from miniNN import Sequential, Linear, MSELoss, Relu, Tanh
from helper import data_generator, train_model


# Setting up hyperparameters
lr = 1e-2
mini_batch_size = 200
nb_epochs = 250

# Define Network
modules = [Linear(2, 25), Relu(), Linear(25, 25), Relu(), Linear(25, 25), Relu(), Linear(25, 2), Tanh()]
model = Sequential(modules)
criterion = MSELoss()

# Generate train set and test set
train_input, train_target, test_input, test_target = data_generator(ratio=0.8, normalized=True)

# Train model
train_model(model,criterion ,train_input, train_target, test_input, test_target, nb_epochs, lr, mini_batch_size)
