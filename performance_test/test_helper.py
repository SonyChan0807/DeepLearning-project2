import torch
from torch import nn
from torch import optim
from miniNN import MSELoss
from memory_profiler import profile
from helper import compute_nb_errors


@profile
def train_model_test(model, train_input, train_target, test_input, test_target, nb_epochs, lr, mini_batch_size, package):

    if package == "miniNN":
        criterion = MSELoss()
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
            test_errors_rate = compute_nb_errors(model, test_input, test_target, test_input.size(0) ) / test_input.size(0) * 100

            print('{:4d}/{}: total_train_loss: {:.04f} train_error {:.02f}% test_error {:.02f}%'.format(e + 1,
                                            nb_epochs,total_loss, train_errors_rate, test_errors_rate))
    else:
        for e in range(0, nb_epochs):
            criterion = nn.MSELoss()
            optimizer = optim.SGD(model.parameters(), lr)
            total_loss = 0

            for b in range(0, train_input.size(0), mini_batch_size):
                output = model(train_input.narrow(0, b, mini_batch_size))
                loss = criterion(output, train_target.narrow(0, b, mini_batch_size))
                total_loss += loss.data[0]
                model.zero_grad()
                loss.backward()
                optimizer.step()

            train_errors_rate = compute_nb_errors_torch(model, train_input, train_target,
                                                      mini_batch_size) / train_input.size(0) * 100
            test_errors_rate = compute_nb_errors_torch(model, test_input, test_target,
                                                     test_input.size(0)) / test_input.size(0) * 100

            print('{:4d}/{}: total_train_loss: {:.04f} train_error {:.02f}% test_error {:.02f}%'.format(e + 1,
                                                    nb_epochs, total_loss, train_errors_rate,test_errors_rate))

def compute_nb_errors_torch(model, data_input, data_target, mini_batch_size):
    nb_data_errors = 0
    for b in range(0, data_input.size(0), mini_batch_size):
        output = model(data_input.narrow(0, b, mini_batch_size))
        _, predicted_classes = torch.max(output.data, 1)
        for k in range(0, mini_batch_size):
            if data_target.data[b + k][predicted_classes[k]] < 0:
                nb_data_errors = nb_data_errors + 1
    return nb_data_errors