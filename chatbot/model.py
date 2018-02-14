
"""
Model to predict the next sentence given an input sequence

"""

import tensorflow as tf

from chatbot.textdata import Batch


class ProjectionOp:

    def __init__(self, shape, scope=None, dtype=None):

        assert len(shape) == 2

        self.scope = scope

    
        with tf.variable_scope('weights_' + self.scope):
            self.W_t = tf.get_variable(
                'weights',
                shape,
                dtype=dtype
            )
            self.b = tf.get_variable(
                'bias',
                shape[0],
                initializer=tf.constant_initializer(),
                dtype=dtype
            )
            self.W = tf.transpose(self.W_t)

    def getWeights(self):

        return self.W, self.b

    def __call__(self, X):

        with tf.name_scope(self.scope):
            return tf.matmul(X, self.W) + self.b


class Model:

    def __init__(self, args, textData):
    
        print("Model creation...")

        self.textData = textData  
        self.args = args  
        self.dtype = tf.float32


        self.encoderInputs  = None
        self.decoderInputs  = None  
        self.decoderTargets = None
        self.decoderWeights = None  


        self.lossFct = None
        self.optOp = None
        self.outputs = None  


        self.buildNetwork()









    
