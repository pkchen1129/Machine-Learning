import time, os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from layers import *
from solver import Solver
from softmax import SoftmaxClassifier
from cnn import ConvNet 
import pdb

def main():
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # pdb.set_trace()
    # data preprocessing for neural network with fully-connected layers2222
    data = {
        'X_train': np.array(x_train[:55000], np.float32).reshape((55000, -1)), # training data
        'y_train': np.array(y_train[:55000], np.int32), # training labels
        'X_val': np.array(x_train[55000:], np.float32).reshape((5000, -1)), # validation data
        'y_val': np.array(y_train[55000:], np.int32), # validation labels
    }
    model = SoftmaxClassifier(hidden_dim=100)

    # data preprocessing for neural network with convolutional layers333333
    # data = {
    #    'X_train': np.array(x_train[:5000], np.float32).reshape((5000, 1,  28, 28)), # training data
    #    'y_train': np.array(y_train[:5000], np.int32), # training labels
    #    'X_val': np.array(x_train[55000:], np.float32).reshape((5000), 1, 28, 28), # validation data
    #    'y_val': np.array(y_train[55000:], np.int32), # validation labels
    # }
    # model = ConvNet(hidden_dim=100)
    # running experiments with convolutional neural network could be time-consuming
    # you may use a small number of training samples for debugging
    # and then take use of all training data to report your experimental results. 
    solver = Solver(model, data,
                    update_rule='sgd',
                    optim_config={'learning_rate': 1e-3,},
                    lr_decay=0.95,
                    num_epochs=10, batch_size=100,
                    print_every=10)
    solver.train()
   
    # Plot the training losses
    plt.plot(solver.loss_history)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training loss history')
    plt.show()
    plt.savefig('loss.png')
    plt.close()

    test_acc = solver.check_accuracy(X=np.array(x_test, np.float32).reshape((10000, -1)), y=y_test)
    # test_acc = solver.check_accuracy(X=np.array(x_test, np.float32).reshape((10000, 1, 28, 28)), y=y_test)
    print('Test accuracy', test_acc)

if __name__== "__main__":
    main()
