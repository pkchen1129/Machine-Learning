from builtins import range
import numpy as np
import math
import matplotlib.pyplot as plt

def fc_forward(x, w, b):
    """
    Computes the forward pass for a fully-connected layer.
    The input x has shape (N, d_in) and contains a minibatch of N
    examples, where each example x[i] has d_in element.
    Inputs:
    - x: A numpy array containing input data, of shape (N, d_in)
    - w: A numpy array of weights, of shape (d_in, d_out)
    - b: A numpy array of biases, of shape (d_out,)
    Returns a tuple of:
    - out: output, of shape (N, d_out)
    - cache: (x, w, b)
    """
    out = None
    # b = np.array([b])
    ###########################################################################
    # TODO: Implement the forward pass. Store the result in the variable out. #
    ###########################################################################
    out = x @ w + b

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def fc_backward(dout, cache):
    """
    Computes the backward pass for a fully_connected layer.
    Inputs:
    - dout: Upstream derivative, of shape (N, d_out)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_in)
      - w: Weights, of shape (d_in, d_out)
      - b: Biases, of shape (d_out,)
    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d_in)
    - dw: Gradient with respect to w, of shape (d_in, d_out)
    - db: Gradient with respect to b, of shape (d_out,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    dx = dout @ w.T # (N ,Dout) * (Din , Dout)' = (N, Din)
    dw = x.T @ dout #  (N , Din)' * (N , Dout)  = (Din, Dout) 
    db = np.sum(dout, axis = 0) #shape would be (dout,0)
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).
    Input:
    - x: Inputs, of any shape
    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    out = np.maximum(0 , x)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).
    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout
    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    out = np.maximum(0,x)
    out[out > 0] = 1
    dx = out * dout 
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def conv_forward(x, w):
    """
    The input consists of N data points, each with C channels, height H and
    width W. We filter each input with F different filters, where each filter
    spans all C channels and has height H' and width W'. 
    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, H', W')
    Returns a tuple of:
    - out: Output data, of shape (N, F, HH, WW) where H' and W' are given by
      HH = H - H' + 1
      WW = W - W' + 1
    - cache: (x, w)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass based on the definition  #
    # of Y in Q1(c).                                                          #
    ###########################################################################
    N, C, H, W = x.shape
    F, C, H_p, W_p = w.shape
    HH = H - H_p + 1##need to add one
    WW = W - W_p + 1
    out = np.zeros((N, F, HH, WW))
    for nn in range(N):
        for ff in range(F):
            for hh in range(HH):
                for ww in range(WW):
                    out[nn,ff,hh,ww] = np.sum(x[nn, :, hh : hh+H_p, ww : ww+W_p] * w[ff,:,:,:])

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w)
    return out, cache


def conv_backward(dout, cache):
    """
    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w) as in conv_forward
    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    """
    dx, dw = None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    x, w = cache
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    N, F, Hh, Ww = dout.shape

    dx = np.zeros_like(x)
    dw = np.zeros_like(w)

    for nn in range(N):
        for ff in range(F):
            for hh in range(Hh):
                for ww in range(Ww):
                    dw[ff,...] += x[nn,:,hh:(hh+HH),ww:(ww+WW)] * dout[nn,ff,hh,ww]
                    dx[nn,:,hh:(hh+HH),ww:(ww+WW)] += w[ff,...] * dout[nn,ff,hh,ww]

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw


def max_pool_forward(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.
    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions
    No padding is necessary here. Output size is given by 
    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']
    

    N, C, H, W = x.shape
    HH = ((H - pool_height) // stride) + 1
    WW = ((W - pool_width) // stride) + 1
    out = np.zeros((N, C, HH, WW))
    for nn in range(N):
        for cc in range(C):
            for hh in range(HH):
                for ww in range(WW):
                    out[nn,cc,hh,ww] = np.max(x[nn , cc , hh*stride:hh*stride+pool_height , ww*stride:ww*stride+pool_width])

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward(dout, cache):
    
    """
    https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/pooling_layer.html

    A naive implementation of the backward pass for a max-pooling layer.
    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.
    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    (x, pool_param) = cache
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']
    N, C, H, W = x.shape
    N, C, HH, WW = dout.shape

    dx = np.zeros_like(x)
    for nn in range(N):
        for cc in range(C):
            for hh in range(HH):
                for ww in range(WW):
                      x_pool = x[nn, cc, hh*stride:hh*stride+pool_height , ww*stride:ww*stride+pool_width]
                      mask = (x_pool == np.max(x_pool)) #for every stride, When the block is same as the max value, it'll be set as true(which is then being 1)
                      dx[nn, cc, hh*stride:hh*stride+pool_height , ww*stride:ww*stride+pool_width] += mask * dout[nn, cc, hh,ww]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.
  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C
  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  loss, dx = None, None
  N , C = x.shape
  
  #loss
  probs = np.exp(x - np.max(x, axis = 1, keepdims = True))
  probs /= np.sum(probs, axis = 1, keepdims = True)
  loss = -np.sum(np.log(probs[np.arange(N) , y])) / N
  ## dx
  dx = probs.copy()
#   dx = probs.all()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx
