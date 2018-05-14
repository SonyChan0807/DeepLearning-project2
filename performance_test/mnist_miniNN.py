
import os
import sys

cwd = os.getcwd()
sys.path.append(cwd)
from miniNN import Sequential, Linear, Relu, Tanh
from performance_test import dlc_practical_prologue as prologue
from performance_test.test_helper import train_model_test


# Setting up hyper parameters
lr = 1e-1
mini_batch_size = 200
nb_epochs = 500

# Define Network
modules = [Linear(784, 100), Relu(), Linear(100, 100), Relu(), Linear(100, 100), Relu(), Linear(100, 10), Tanh()]
model = Sequential(modules)

# Split data into train set and test set
train_input, train_target, test_input, test_target = prologue.load_data(one_hot_labels=True,
                                                                        normalize=True)

# Train model
train_model_test(model, train_input, train_target, test_input,
                 test_target, nb_epochs, lr, mini_batch_size, "miniNN")
