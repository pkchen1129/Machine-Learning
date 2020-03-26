import numpy as np

from layers import fc_forward, fc_backward
from rnn_layers import *

class CaptioningRNN(object):
    """
    A CaptioningRNN produces captions from image features using a recurrent
    neural network.
    The RNN receives input vectors of size D, has a vocab size of V, works on
    sequences of length T, has an RNN hidden dimension of H, uses word vectors
    of dimension W, and operates on minibatches of size N.
    Note that we don't use any regularization for the CaptioningRNN.
    """

    def __init__(self, word_to_idx, input_dim=512, wordvec_dim=128,
                 hidden_dim=128, cell_type='rnn', dtype=np.float32):
        """
        Construct a new CaptioningRNN instance.
        Inputs:
        - word_to_idx: A dictionary giving the vocabulary. It contains V entries,
          and maps each string to a unique integer in the range [0, V).
        - input_dim: Dimension D of input image feature vectors.
        - wordvec_dim: Dimension W of word vectors.
        - hidden_dim: Dimension H for the hidden state of the RNN.
        - cell_type: What type of RNN to use; either 'rnn' or 'lstm'.
        - dtype: numpy datatype to use; use float32 for training and float64 for
          numeric gradient checking.
        """
        if cell_type not in {'rnn', 'lstm'}:
            raise ValueError('Invalid cell_type "%s"' % cell_type)

        self.cell_type = cell_type
        self.dtype = dtype
        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.items()}
        self.params = {}

        vocab_size = len(word_to_idx)

        self._null = word_to_idx['<NULL>']
        self._start = word_to_idx.get('<START>', None)
        self._end = word_to_idx.get('<END>', None)

        # Initialize word vectors
        self.params['W_embed'] = np.random.randn(vocab_size, wordvec_dim)
        self.params['W_embed'] /= 100

        # Initialize CNN -> hidden state projection parameters
        self.params['W_proj'] = np.random.randn(input_dim, hidden_dim)
        self.params['W_proj'] /= np.sqrt(input_dim)
        self.params['b_proj'] = np.zeros(hidden_dim)

        # Initialize parameters for the RNN
        self.params['Wx'] = np.random.randn(wordvec_dim, hidden_dim)
        self.params['Wx'] /= np.sqrt(wordvec_dim)
        self.params['Wh'] = np.random.randn(hidden_dim, hidden_dim)
        self.params['Wh'] /= np.sqrt(hidden_dim)
        self.params['b'] = np.zeros(hidden_dim)

        # Initialize output to vocab weights
        self.params['W_vocab'] = np.random.randn(hidden_dim, vocab_size)
        self.params['W_vocab'] /= np.sqrt(hidden_dim)
        self.params['b_vocab'] = np.zeros(vocab_size)

        # Cast parameters to correct dtype
        for k, v in self.params.items():
            self.params[k] = v.astype(self.dtype)


    def loss(self, features, captions):
        """
        Compute training-time loss for the RNN. We input image features and
        ground-truth captions for those images, and use an RNN to compute
        loss and gradients on all parameters.
        Inputs:
        - features: Input image features, of shape (N, D)
        - captions: Ground-truth captions; an integer array of shape (N, T) where
          each element is in the range 0 <= y[i, t] < V
        Returns a tuple of:
        - loss: Scalar loss
        - grads: Dictionary of gradients parallel to self.params
        """
        # Cut captions into two pieces: captions_in has everything but the last word
        # and will be input to the RNN; captions_out has everything but the first
        # word and this is what we will expect the RNN to generate. These are offset
        # by one relative to each other because the RNN should produce word (t+1)
        # after receiving word t. The first element of captions_in will be the START
        # token, and the first element of captions_out will be the first word.
        captions_in = captions[:, :-1]
        captions_out = captions[:, 1:]

        # You'll need this
        mask = (captions_out != self._null)

        # Weight and bias for the affine transform from image features to initial
        # hidden state
        W_proj, b_proj = self.params['W_proj'], self.params['b_proj']

        # Word embedding matrix
        W_embed = self.params['W_embed']

        # Input-to-hidden, hidden-to-hidden, and biases for the RNN
        Wx, Wh, b = self.params['Wx'], self.params['Wh'], self.params['b']

        # Weight and bias for the hidden-to-vocab transformation.
        W_vocab, b_vocab = self.params['W_vocab'], self.params['b_vocab']

        loss, grads = 0.0, {}
        batch_size, input_dim = features.shape
        _, n_time_steps = captions_in.shape
        wordvec_dim = Wx.shape[0]
        hidden_dim = Wh.shape[0]
        vocab_size = W_vocab.shape[1]
        ############################################################################
        # TODO: Implement the forward and backward passes for the CaptioningRNN.   #
        # In the forward pass you will need to do the following:                   #
        # (1) Use an fc transformation to compute the initial hidden state         #
        #     from the image features. This should produce an array of shape (N, H)#
        # (2) Use a word embedding layer to transform the words in captions_in     #
        #     from indices to vectors, giving an array of shape (N, T, W).         #
        # (3) Use a vanilla RNN to process the sequence of input word vectors      #
        #     and produce hidden state vectors for all timesteps, producing        #
        #     an array of shape (T, N, H).                                         #
        # (4) Use a (temporal) fc transformation to compute scores over the        #
        #     vocabulary at every timestep using the hidden states, giving an      #
        #     array of shape (N, T, V).                                            #
        # (5) Use (temporal) softmax to compute loss using captions_out, ignoring  #
        #     the points where the output word is <NULL> using the mask above.     #
        #                                                                          #
        # In the backward pass you will need to compute the gradient of the loss   #
        # with respect to all model parameters. Use the loss and grads variables   #
        # defined above to store loss and gradients; grads[k] should give the      #
        # gradients for self.params[k].                                            #
        ############################################################################

        # Forward Pass
                         # N x T x D
         # (1) compute the initial hidden state (N, H)
        h0, cache_h0 = fc_forward(features, W_proj, b_proj)

        # (2) transform the words in captions_in to vectors (N, T, W)
        x, cache_emb = word_embedding_forward(captions_in, W_embed)
        x_trans = np.transpose( x , (1,0,2))
        # (3) produce hidden state vectors for all timestapes (N, T, H)
        h_trans, cache_h = rnn_forward(x_trans, h0, Wx, Wh, b)
        h = np.transpose( h_trans , (1,0,2))
        # (4) compute scores over the vocabulary (N, T, V)
        out, cache_out = temporal_fc_forward(h, W_vocab, b_vocab)

        # (5) compute softmax loss using captions_out
        loss, dout = temporal_softmax_loss(out, captions_out, mask)
        
        
        
        
        # Gradients####################################
        # (6) backprop for (4)
        dout = dout.reshape(-1, vocab_size)   # (N x T, V)
        dh, grads['W_vocab'], grads['b_vocab'] = temporal_fc_backward(dout, cache_out)
        dh = np.transpose( dh , (1,0,2))
        # (7) backprop for (3)
        dx, dh0, grads['Wx'], grads['Wh'], grads['b'] = rnn_backward(dh, cache_h)
        dx = np.transpose(dx,(1,0,2))
        # (8) backprop for (2)
        grads['W_embed'] =  word_embedding_backward(dx, cache_emb)

        # (9) backprop for (1)
        _, grads['W_proj'], grads['b_proj'] = fc_backward(dh0, cache_h0) 

        


        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


    def sample(self, features, max_length=30):
        """
        Run a test-time forward pass for the model, sampling captions for input
        feature vectors.
        At each timestep, we embed the current word, pass it and the previous hidden
        state to the RNN to get the next hidden state, use the hidden state to get
        scores for all vocab words, and choose the word with the highest score as
        the next word. The initial hidden state is computed by applying an affine
        transform to the input image features, and the initial word is the <START>
        token.
        For LSTMs you will also have to keep track of the cell state; in that case
        the initial cell state should be zero.
        Inputs:
        - features: Array of input image features of shape (N, D).
        - max_length: Maximum length T of generated captions.
        Returns:
        - captions: Array of shape (N, max_length) giving sampled captions,
          where each element is an integer in the range [0, V). The first element
          of captions should be the first sampled word, not the <START> token.
        """
        N = features.shape[0]
        captions = self._null * np.ones((N, max_length), dtype=np.int32)

        # Unpack parameters
        W_proj, b_proj = self.params['W_proj'], self.params['b_proj']
        W_embed = self.params['W_embed']
        Wx, Wh, b = self.params['Wx'], self.params['Wh'], self.params['b']
        W_vocab, b_vocab = self.params['W_vocab'], self.params['b_vocab']

        _, input_dim = features.shape
        wordvec_dim = Wx.shape[0]
        hidden_dim = Wh.shape[0]
        vocab_size = W_vocab.shape[1]
        ###########################################################################
        # TODO: Implement test-time sampling for the model. You will need to      #
        # initialize the hidden state of the RNN by applying the learned affine   #
        # transform to the input image features. The first word that you feed to  #
        # the RNN should be the <START> token; its value is stored in the         #
        # variable self._start. At each timestep you will need to do to:          #
        # (1) Embed the previous word using the learned word embeddings           #
        # (2) Make an RNN step using the previous hidden state and the embedded   #
        #     current word to get the next hidden state.                          #
        # (3) Apply the learned fc transformation to the next hidden state to     #
        #     get scores for all words in the vocabulary                          #
        # (4) Select the word with the highest score as the next word, writing it #
        #     to the appropriate slot in the captions variable                    #
        #                                                                         #
        # For simplicity, you do not need to stop generating after an <END> token #
        # is sampled, but you can if you want to.                                 #
        #                                                                         #
        # HINT: You will not be able to use the rnn_forward functions; you'll     # 
        # need to call rnn_step_forward or lstm_step_forward in a loop.           #
        ###########################################################################
        h, _ = fc_forward(features, W_proj, b_proj)
        c = 0
        for t in range(max_length):
            #(1) Embed the previous word using the learned word embeddings  
            if t==0:
                word_vec = np.zeros((N, wordvec_dim))
            else:
                word_vec, _ = word_embedding_forward(word_idx, W_embed)
            # (2) RNN step
            h, _ = rnn_step_forward(word_vec.reshape(-1, wordvec_dim), h, Wx, Wh, b)  

            # (3) get socres for all words in the vocab
            out, _ = fc_forward(h, W_vocab, b_vocab)     # (N, V)

            # (4) select word with highest score 
            word_idx = np.argmax(out, axis=1).reshape(-1, 1)     # (N, 1)
            captions[:, t:t+1] = word_idx

            if self.idx_to_word == '<END>':
                break

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        return captions
