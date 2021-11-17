# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
_LAYER_UIDS = {}


def uniform(shape, scale=0.05, name=None):
    """Uniform init."""
    initial = tf.random_uniform(shape, minval=-scale, maxval=scale, dtype=tf.float32)
    return tf.Variable(initial, name=name)
def zeros(shape, name=None):
    """All zeros."""
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)
def ones(shape, name=None):
    """All ones."""
    initial = tf.ones(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)
def glorot(shape, name=None):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)
def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name = name)
def bias_variable(shape, name):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial, name = name)
def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)


def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res

def GCN_dot(support, x, trte_idx):
    result=tf.zeros([1,np.shape(support)[0]+1])
    for i in range(np.shape(support)[0]):
        t1=tf.constant([[0.0]])
        for i1 in range(np.shape(support)[0]):
            t2=tf.constant([[0]])
            for j in range(np.shape(support)[0]):
                t2=tf.concat([t1,tf.reshape(support[i,j]*x[j,i1],[1,1])],0)
            t2=tf.reshape(tf.reduce_sum(t2),[1,1])
            t1=tf.concat([t1,t2],0)
        result=tf.concat([result,t1],1)
    return result[1:np.shape(support)[0]+1,1:np.shape(support)[0]]
    

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


class CNNHSI(object):
    def __init__(self, dropout = 0, act = tf.nn.softplus, filter1 = [], dim = 0):
        self.act = act
        self.filter1 = filter1
        self.dim = dim

        if dropout!=0:
            self.dropout = dropout
        else:
            self.dropout = 0.
        self.vars={}
        self.vars['W_conv'] = zeros(self.filter1, 'CNNweight_0') #
        self.vars['b_conv'] = bias_variable([1], 'CNNbias_0')
    def __call__(self, inputs):
        outputs = self._call(inputs)
        return outputs
    def _call(self, inputs):   

        x = inputs
        h_conv1 = self.act(conv2d(x[:,:,:,0:1], self.vars['W_conv']) + self.vars['b_conv'])
        for i in range(1, self.dim):
            h_conv1 = tf.concat([h_conv1, self.act(conv2d(x[:,:,:,i:i+1], self.vars['W_conv']) + self.vars['b_conv'])],3)

        return h_conv1
    
class SoftmaxLayer(object):
    def __init__(self, input_num, output_num, dropout = 0, act = tf.nn.softplus, bias = True ):
        self.bias = bias
        self.act = act
        self.output_num = output_num
        self.input_num = input_num
        self.vars={}
        self.vars['weights'] = glorot(shape = [self.input_num, self.output_num], name = 'weight_0')
        self.vars['bias'] = uniform(shape = [self.output_num], name = 'bias_0')
#        self.is_sparse = is_sparse
    def __call__(self, inputs):
        outputs = self._call(inputs)
        return outputs
    def _call(self, inputs):
        x = inputs
        pre_sup = tf.matmul(x, self.vars['weights'])
        return self.act(pre_sup + self.vars['bias'])
  
class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    Implementation inspired by keras (http://keras.io).

    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])
      
class GraphConvolution(Layer):
    """Graph convolution layer."""
    def __init__(self, input_dim, output_dim, support, num_features_nonzero, act=tf.nn.softplus, bias=False,
                 sparse_inputs=False, isnorm=False, isSparse = False, dropout = 0,
                 **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)
        self.act = act
        self.support = support#DAD
        self.bias = bias
        self.isnorm = isnorm
        self.isSparse = isSparse
        self.sparse_inputs = sparse_inputs
        self.dropout = dropout
        self.num_features_nonzero = num_features_nonzero
        # helper variable for sparse dropout

        with tf.variable_scope(self.name + '_vars'):
            for i in range(1):
                self.vars['weights_' + str(i)] = glorot([input_dim, output_dim],
                                                        name='weights_' + str(i))
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs
        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)


        # convolve
        supports = list()
        pre_sup = dot(x, self.vars['weights_' + str(0)], sparse = self.sparse_inputs)
        support = dot( self.support, pre_sup, sparse=self.isSparse )


        
        supports.append(support)
        output = tf.add_n(supports)

        # bias
        if self.bias:
            output += self.vars['bias']
        if self.isnorm==True:
            output = tf.nn.l2_normalize(output, dim=0)
        return self.act(output)
    
class MLP(Layer):
    """Graph convolution layer."""
    def __init__(self, input_dim, output_dim, act=tf.nn.softplus, bias=False,
                 sparse_inputs=False, isnorm=False, isSparse = False,   **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.act = act
        self.bias = bias
        self.isnorm = isnorm
        self.isSparse = isSparse
        self.sparse_inputs = sparse_inputs
        # helper variable for sparse dropout

        with tf.variable_scope(self.name + '_vars'):
            for i in range(1):
                self.vars['weights_' + str(i)] = glorot([input_dim, output_dim],
                                                        name='weights_' + str(i))
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs
#        x = tf.nn.dropout(x, 1)
        # convolve
        supports = list()
        support = dot(x, self.vars['weights_' + str(0)], sparse = self.sparse_inputs)

        
        supports.append(support)
        output = tf.add_n(supports)

        # bias
        if self.bias:
            output += self.vars['bias']
        if self.isnorm==True:
            output = tf.nn.l2_normalize(output, dim=0)
        return self.act(output)    