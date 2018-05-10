from torch.autograd import Variable
from torch import nn
from helper import data_generator, train_model_test

# Setting up hyper parameters
lr = 1e-2
mini_batch_size = 200
nb_epochs = 250

# Split data into train set and test set
train_input, train_target, test_input, test_target = data_generator(ratio=0.8, normalized=True)

train_input, train_target = Variable(train_input), Variable(train_target)
test_input, test_target = Variable(test_input), Variable(test_target)

# Define Network
model = nn.Sequential(
    nn.Linear(2, 25),
    nn.ReLU(),
    nn.Linear(25, 25),
    nn.ReLU(),
    nn.Linear(25, 25),
    nn.ReLU(),
    nn.Linear(25, 2),
    nn.Tanh()
)

# Train model
train_model_test(model, train_input, train_target, test_input,
                 test_target, nb_epochs, lr, mini_batch_size, "pytorch")
