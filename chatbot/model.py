
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

    def buildNetwork(self):

        outputProjection = None
        if 0 < self.args.softmaxSamples < self.textData.getVocabularySize():
            outputProjection = ProjectionOp(
                (self.textData.getVocabularySize(), self.args.hiddenSize),
                scope='softmax_projection',
                dtype=self.dtype
            )

            def sampledSoftmax(labels, inputs):
                labels = tf.reshape(labels, [-1, 1])  
                localWt     = tf.cast(outputProjection.W_t,             tf.float32)
                localB      = tf.cast(outputProjection.b,               tf.float32)
                localInputs = tf.cast(inputs,                           tf.float32)

                return tf.cast(
                    tf.nn.sampled_softmax_loss(
                        localWt,  
                        localB,
                        labels,
                        localInputs,
                        self.args.softmaxSamples,  
                        self.textData.getVocabularySize()), 
                    self.dtype)








    
