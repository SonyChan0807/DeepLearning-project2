
import torch
import os
import sys

cwd = os.getcwd()
sys.path.append(cwd)
from miniNN import Sequential, Linear, MSELoss, Relu, Tanh
from helper import data_generator


def train_model(model, train_input, train_target, test_input, test_target, nb_epochs, lr, mini_batch_size):

    criterion = MSELoss()

    train_errors_list = []
    test_errors_list = []

    for m in range(nb_epochs):
        total_loss = 0
        for b in range(0, train_input.size(0), mini_batch_size):
            output = model.forward(train_input.narrow(0, b, mini_batch_size))
            total_loss += criterion.forward(output, train_target.narrow(0, b, mini_batch_size))
            model.zero_gradient()
            dloss = criterion.backward()
            model.backward(dloss)
            model.update_params(lr)

        train_errors_rate = compute_nb_errors(model, train_input, train_target,mini_batch_size ) / train_input.size(0) * 100
        test_errors_rate = compute_nb_errors(model, test_input, test_target, test_input.size(0) ) / test_input.size(0) * 100

        train_errors_list.append(train_errors_rate)
        test_errors_list.append(test_errors_rate)

        print('{:4d}/{}: total_train_loss: {:.04f} train_error {:.02f}% test_error {:.02f}%'.format(m + 1,
                                        nb_epochs,total_loss, train_errors_rate, test_errors_rate))

    return train_errors_list, test_errors_list


def compute_nb_errors(model, data_input, data_target, mini_batch_size ):
    nb_data_errors = 0
    for b in range(0, data_input.size(0), mini_batch_size):
        output = model.forward(data_input.narrow(0, b, mini_batch_size))
        _, predicted_classes = torch.max(output, 1)
        for k in range(0, mini_batch_size):
            if data_target[b + k][predicted_classes[k]] < 0:
                nb_data_errors = nb_data_errors + 1

    return nb_data_errors



if __name__ == "__main__":

    # Setting up hyper parameters
    lr = 1e-2
    mini_batch_size = 200
    nb_epochs = 1000

    # Define Network
    modules = [Linear(2, 25), Relu(), Linear(25, 25), Relu(), Linear(25, 25), Relu(), Linear(25, 2), Tanh()]
    model = Sequential(modules)

    # Split data into train set and test set
    train_input, train_target, test_input, test_target = data_generator(ratio=0.8, normalized=True)

    # Train model
    train_errors_list, test_errors_list = train_model(model, train_input, train_target, test_input,
                                                            test_target, nb_epochs, lr, mini_batch_size)