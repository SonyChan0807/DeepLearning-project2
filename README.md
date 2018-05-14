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
    |   └──sequential.py              : Containing  Sequential
    scripts
    |   ├── dlc_practical_prologue.py  : Utility for loading Mnist
    |   ├── disc_miniNN.py             : Script to run miniNN on Disc
    |   ├── disc_pytorch.py            : Script to run PyTorch on Disc
    |   ├── mnist_miniNN.py            : Script to run miniNN on Mnist 
    |   ├── mnist_pytorch.py           : Script to run PyTorch on Mnist
    |   └── test_helper.py             : helper function including @profile decorator 
    ├── Report.pdf                     : Report
    ├── test.py                        : Main script to test miniNN
    ├── helper.py                      : helper functions for test.py
    ├── performance_test               : Directory containing all test 
    └── README.md                      : The README guideline and explanation for our project.

# Usage
------------

#### Test miniNN 
To run our best model, simply `$ git clone` the repository then run `$ python test.py` 


# Performance Test
------------
## Additional Library
Before running performance test, install memory_profiler by `$ pip install install memory_profiler`, check here for reference 
[memory_profiler](https://pypi.org/project/memory_profiler/) 

## Rung the script
Go to the DeepLearning-project2 directory and run 
`$ mprof run --include-children performance_test/mnist_miniNN.py` and `*.dat` file will generated in the folder.

## Plot the result
`$ mprof plot *.dat`

