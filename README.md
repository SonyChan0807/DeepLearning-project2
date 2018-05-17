# Design a mini deep learning framework

We implement PyTorch like modules in this project. Details can be found in the attached report.

# Contents
------------
  * [Requirements](#requirements)
  * [Brief Project Structure](#brief-project-structure)
  * [Usage](#usage)
  * [Performance Test](#performance-test)


# Requirements
------------
## For Windows

  * Windows 10
  * [Anaconda](https://www.anaconda.com/download/) with Python 3
  * [PyTorch](https://anaconda.org/peterjc123/pytorch) 0.3.1
  
## For Linux and macOS
  * Ubuntu 16.04 LTS / macOS High Sierra version 10.13.4
  * [Python](https://www.python.org/downloads/) 3.6.5
  * [PyTorch](https://pytorch.org/) 0.3.1 without GPU

# Brief Project Structure
------------

    ├── miniNN                         : Directory containing all implemented modules
    |   ├── __init__.py                : Declare the package
    |   ├── module.py                  : Containing Module 
    |   ├── loss.py                    : Containing Loss, MSELoss
    |   ├── activation.py              : Containing ReLu, Tanh
    |   ├── linear.py                  : Containing Linear
    |   └──sequential.py               : Containing  Sequential 
    |
    ├── performance_test               : Directory containing all test scripts
    |   ├── dlc_practical_prologue.py  : Utility for loading Mnist
    |   ├── disc_miniNN.py             : Script to run miniNN on Disc
    |   ├── disc_pytorch.py            : Script to run PyTorch on Disc
    |   ├── mnist_miniNN.py            : Script to run miniNN on Mnist 
    |   ├── mnist_pytorch.py           : Script to run PyTorch on Mnist
    |   └── test_helper.py             : helper function anotated by @profile decorator 
    |
    ├── Report.pdf                     : Report
    ├── test.py                        : Main script to test miniNN
    ├── helper.py                      : helper functions for test.py
    └── README.md                      : The README guideline and explanation for our project.

# Usage
------------

## Sample code
To run our sample code, simply `$ git clone` the repository then run `$ python test.py`. Below is an explanation of the contents of `test.py`.

## Explanation
We first import the components and loss function that we will be using. Note that the full list of available components are listed in.

```python
from miniNN import Sequential, Linear, Relu, Tanh # deep net components
from miniNN import MSELoss                        # loss function
```

The architecture can then be specified by intializing the components sequentially in a python list. Shown below is an example to create a network with 2 input neurons, 3 hidden layers with 25 units each and ReLu activation functions, and a final output layer with 2 classes and a Tanh activation function. The `sequential` function takes in the list and 

```python
modules = [Linear(2, 25), Relu(), Linear(25, 25), Relu(), Linear(25, 25), Relu(), Linear(25, 2), Tanh()]
model = Sequential(modules)
```

We then specify the objective that the model will try to minimize. Here it is the mean squared error.

```python
criterion = MSELoss()
```

Then by passing the relevant arguments into `train_model`. The `data_generator` used here simply creates 

```python
# Generate train set and test set
train_input, train_target, test_input, test_target = data_generator(ratio=0.8, normalized=True)

# Training parameters
learning_rate = 1e-2
mini_batch_size = 200
nb_epochs = 250

# Train model
train_model(model, criterion, train_input, train_target, 
                              test_input,  test_target, 
                              nb_epochs,   learning_rate, mini_batch_size)
```

During training, the script outputs the training and validation XXX at each epoch. To avoid clutter, we show only the output at the final epoch.

`Insert output here`

Once the model is trained, classification can simply be done by calling the `forward` method of the model object.

```python
prediction = model.forward(test_input)
```

# Performance Test
------------
## Additional Library
Before running performance test, install memory_profiler by `$ pip install install memory_profiler`, check here for reference 
[memory_profiler](https://pypi.org/project/memory_profiler/) 

## Rung the script
Go to the DeepLearning-project2 directory and run 
`$ mprof run --include-children performance_test/mnist_miniNN.py` and `*.dat` file will generated in the folder.

## Plot the result
 Run `$ mprof plot *.dat` in the same directory
### The example of the plot 
![Alt text](https://github.com/SonyChan0807/DeepLearning-project2/blob/save_images/img/miniNN-minst.png)
