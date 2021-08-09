from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    num_train=X.shape[0]
    num_class=W.shape[1]
    loss = 0.0
    dW = np.zeros_like(W)
    ############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    fx_softmax=np.dot(X,W)
    for i in range(fx_softmax.shape[0]):
        fx_softmax[i]=np.exp(fx_softmax[i])
        total=np.sum(fx_softmax[i])
        fx_softmax[i]=fx_softmax[i]/total
        loss+=-np.log(fx_softmax[i][y[i]])
        S_Y=fx_softmax[i]
        for j in range(W.shape[1]):
            if j==y[i]:
                dW[:,j]+=(S_Y[j]-1)*(X[i].T)
            else:
                dW[:,j]+=S_Y[j]*(X[i].T)
    loss/=num_train
    loss+=0.5*reg*np.sum(W**2)
    dW=dW/num_train+reg*W
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    fx_softmax=np.exp(np.dot(X,W))
    total=np.sum(fx_softmax,axis=1)
    total=np.reshape(total,(total.shape[0],1))
    fx_softmax=fx_softmax/total
    loss-=np.sum(np.log(fx_softmax[np.arange(0,X.shape[0],1),y]))
    loss/=X.shape[0]
    loss+=0.5*reg*np.sum(W**2)
    fx_softmax[np.arange(0,X.shape[0],1),y]-=1
    dW=np.dot(X.T,fx_softmax)
    dW=dW/X.shape[0]+reg*W
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
