
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

  
        def create_rnn_cell():
            encoDecoCell = tf.contrib.rnn.BasicLSTMCell(  
                self.args.hiddenSize,
            )
            if not self.args.test:  
                encoDecoCell = tf.contrib.rnn.DropoutWrapper(
                    encoDecoCell,
                    input_keep_prob=1.0,
                    output_keep_prob=self.args.dropout
                )
            return encoDecoCell
        encoDecoCell = tf.contrib.rnn.MultiRNNCell(
            [create_rnn_cell() for _ in range(self.args.numLayers)],
        )


        with tf.name_scope('placeholder_encoder'):
            self.encoderInputs  = [tf.placeholder(tf.int32,   [None, ]) for _ in range(self.args.maxLengthEnco)]  

        with tf.name_scope('placeholder_decoder'):
            self.decoderInputs  = [tf.placeholder(tf.int32,   [None, ], name='inputs') for _ in range(self.args.maxLengthDeco)] 
            self.decoderTargets = [tf.placeholder(tf.int32,   [None, ], name='targets') for _ in range(self.args.maxLengthDeco)]
            self.decoderWeights = [tf.placeholder(tf.float32, [None, ], name='weights') for _ in range(self.args.maxLengthDeco)]

        
        decoderOutputs, states = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(
            self.encoderInputs,  
            self.decoderInputs,  
            encoDecoCell,
            self.textData.getVocabularySize(),
            self.textData.getVocabularySize(),  
            embedding_size=self.args.embeddingSize,  
            output_projection=outputProjection.getWeights() if outputProjection else None,
            feed_previous=bool(self.args.test)  
        )


        if self.args.test:
            if not outputProjection:
                self.outputs = decoderOutputs
            else:
                self.outputs = [outputProjection(output) for output in decoderOutputs]

           
        else:

            self.lossFct = tf.contrib.legacy_seq2seq.sequence_loss(
                decoderOutputs,
                self.decoderTargets,
                self.decoderWeights,
                self.textData.getVocabularySize(),
                softmax_loss_function= sampledSoftmax if outputProjection else None  
            )
            tf.summary.scalar('loss', self.lossFct) 

            opt = tf.train.AdamOptimizer(
                learning_rate=self.args.learningRate,
                beta1=0.9,
                beta2=0.999,
                epsilon=1e-08
            )
            self.optOp = opt.minimize(self.lossFct)

    def step(self, batch):

        feedDict = {}
        ops = None

        if not self.args.test:  
            for i in range(self.args.maxLengthEnco):
                feedDict[self.encoderInputs[i]]  = batch.encoderSeqs[i]
            for i in range(self.args.maxLengthDeco):
                feedDict[self.decoderInputs[i]]  = batch.decoderSeqs[i]
                feedDict[self.decoderTargets[i]] = batch.targetSeqs[i]
                feedDict[self.decoderWeights[i]] = batch.weights[i]

            ops = (self.optOp, self.lossFct)
        else: 
            for i in range(self.args.maxLengthEnco):
                feedDict[self.encoderInputs[i]]  = batch.encoderSeqs[i]
            feedDict[self.decoderInputs[0]]  = [self.textData.goToken]

            ops = (self.outputs,)

 
        return ops, feedDict
