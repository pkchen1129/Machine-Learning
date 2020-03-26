import numpy as np
import matplotlib.pyplot as plt
from layers import *


class ConvNet(object):
  """
  A convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - fc - relu - fc - softmax
  You may also consider adding dropout layer or batch normalization layer. 
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  def __init__(self, input_dim=(1, 28, 28), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.

    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.dtype = dtype
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights for the convolutional layer using the keys 'W1' (here      #
    # we do not consider the bias term in the convolutional layer);            #
    # use keys 'W2' and 'b2' for the weights and biases of the                 #
    # hidden fully-connected layer, and keys 'W3' and 'b3' for the weights     #
    # and biases of the output affine layer.                                   #
    ############################################################################
    #Weights should be initialized from a Gaussian with standard deviation equal to weight_scale;
    self.params['W1'] = np.random.normal(scale=weight_scale, size=(num_filters, input_dim[0], filter_size, filter_size))
    W2_row_size = num_filters * int(  ((input_dim[1]- filter_size + 1)/2) * ((input_dim[2]- filter_size + 1)/2) )
    
    self.params['W2'] = np.random.normal(scale=weight_scale, size=(W2_row_size, hidden_dim)) 
    self.params['W3'] = np.random.normal(scale=weight_scale, size=(hidden_dim, num_classes))
    # self.params['b1'] = np.zeros(num_filters)
    self.params['b2'] = np.zeros(hidden_dim)
    self.params['b3'] = np.zeros(num_classes)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.items():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    """
    W1 = self.params['W1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    out_1 , cache_1 = conv_forward(X , W1)
    out_2 , cache_2 = relu_forward(out_1)
    out_3 , cache_3 = max_pool_forward(out_2 , pool_param)

    A, B, C, D = np.shape(out_3)
    flat = out_3.reshape(out_3.shape[0], out_3.shape[1]*out_3.shape[2]*out_3.shape[3])

    out_4 , cache_4 = fc_forward(flat, W2, b2)
    out_5 , cache_5 = relu_forward(out_4)
    out_6 , cache_6 = fc_forward(out_5, W3, b3)

    scores = out_6
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k].                                                      #
    ############################################################################
    loss, dscores = softmax_loss(scores , y)
    # loss += sum(0.5*np.sum(W_tmp**2) for W_tmp in [W1, W2, W3])
    dx_6 , grads['W3'] , grads['b3'] = fc_backward(dscores , cache_6)
    dx_5 = relu_backward(dx_6, cache_5)
    dx_4 , grads['W2'] , grads['b2'] = fc_backward(dx_5 , cache_4)
    d_flat_back = dx_4.reshape(A, B, C, D)
    
    dx_3 = max_pool_backward(d_flat_back, cache_3)
    dx_2 = relu_backward(dx_3, cache_2)
    dx_1 , grads['W1'] = conv_backward(dx_2, cache_1)


    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  

