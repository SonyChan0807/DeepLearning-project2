

import math
import torch
from torch import nn
from torch import optim
from miniNN import MSELoss
# from memory_profiler import profile



def train_model(model,criterion, train_input, train_target, test_input, test_target, nb_epochs, lr, mini_batch_size):

    """Execute the training process of given model

    :param model: Sequential module of miniNN
    :param criterion: Loss function of miniNN
    :param train_input:  tensor of train data
    :param train_target: tensor of train label
    :param test_input:  tensor of  test data
    :param test_target: tensor of test label
    :param nb_epochs: number of epochs
    :param lr: learning rate
    :param mini_batch_size: number of mini-batch

    :return: None
    """

    for e in range(nb_epochs):
        total_loss = 0

        for b in range(0, train_input.size(0), mini_batch_size):

            output = model.forward(train_input.narrow(0, b, mini_batch_size))
            total_loss += criterion.forward(output, train_target.narrow(0, b, mini_batch_size))

            model.zero_gradient()
            dloss = criterion.backward()
            model.backward(dloss)
            model.update_params(lr)

        train_errors_rate = compute_nb_errors(model, train_input, train_target,mini_batch_size ) / train_input.size(0) * 100
        test_errors_rate = compute_nb_errors(model, test_input, test_target, test_input.size(0)) / test_input.size(0) * 100

        print('{:4d}/{}: total_train_loss: {:.04f} train_error {:.02f}% test_error {:.02f}%'.format(e + 1,
                                            nb_epochs,total_loss, train_errors_rate, test_errors_rate))


def compute_nb_errors(model, data_input, data_target, mini_batch_size):
    """ Calculate the prediction errors

    :param model: model: Sequential module of miniNN
    :param data_input:  Tensor of data for prediction
    :param data_target: Tensor of data label
    :param mini_batch_size: number of mini-batch

    :return: number of errors
    """
    nb_data_errors = 0

    for b in range(0, data_input.size(0), mini_batch_size):

        output = model.forward(data_input.narrow(0, b, mini_batch_size))
        _, predicted_classes = torch.max(output, 1)

        for k in range(0, mini_batch_size):
            if data_target[b + k][predicted_classes[k]] < 0:
                nb_data_errors = nb_data_errors + 1

    return nb_data_errors


def data_generator(ratio=0.8, normalized=False):
    """Generate Disc data described in Project2 and split data into train set and test set

    :param ratio: (optional) the ratio of train and test set
    :param normalized: (optional)  normalized the data if set to true

    :return: train_input, train_target, test_input, test_target
    """
    data = torch.FloatTensor(1000, 2).uniform_(0, 1) - 0.5
    distance = torch.sqrt(torch.pow(data[:, 0], 2) + torch.pow(data[:, 1], 2)).view(-1, 1)
    radius = 1 / math.sqrt(2 * math.pi)
    inside = distance.clone().apply_(lambda x: 1 if x < radius else -1)
    outside = distance.clone().apply_(lambda x: 1 if x > radius else -1)
    target = torch.cat((inside, outside), 1)

    if normalized:
        data = (data - data.mean()) / data.std()

    split_idx = int(data.size(0) * ratio)
    end_idx = data.size(0) + 1

    return data[0:split_idx, ], target[0:split_idx, ], data[split_idx:end_idx, ], target[split_idx:end_idx, ]



