
import os
import sys
cwd = os.getcwd()
sys.path.append(cwd)

from torch.autograd import Variable
from torch import nn
from performance_test import dlc_practical_prologue as prologue
from performance_test.test_helper import train_model_test

# Setting up hyper parameters
lr = 1e-1
mini_batch_size = 200
nb_epochs = 500

# Split data into train set and test set
train_input, train_target, test_input, test_target = prologue.load_data(one_hot_labels=True,
                                                                        normalize=True)

train_input, train_target = Variable(train_input), Variable(train_target)
test_input, test_target = Variable(test_input), Variable(test_target)

# Define Network
model = nn.Sequential(
    nn.Linear(784, 100),
    nn.ReLU(),
    nn.Linear(100, 100),
    nn.ReLU(),
    nn.Linear(100, 100),
    nn.ReLU(),
    nn.Linear(100, 10),
    nn.Tanh()
)

# Train model
train_model_test(model, train_input, train_target, test_input,
                 test_target, nb_epochs, lr, mini_batch_size, "pytorch")
