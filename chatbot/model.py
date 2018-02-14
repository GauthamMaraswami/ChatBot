
"""
Model to predict the next sentence given an input sequence

"""

import tensorflow as tf

from chatbot.textdata import Batch


class ProjectionOp:

    def __init__(self, shape, scope=None, dtype=None):

        assert len(shape) == 2

        self.scope = scope

    




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









    
