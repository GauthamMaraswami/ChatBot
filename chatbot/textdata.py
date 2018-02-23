
"""
Loads the dialogue corpus, builds the vocabulary
"""

import numpy as np
import nltk  
from tqdm import tqdm 
import pickle 
import math 
import os  
import random
import string
import collections

from chatbot.corpus.cornelldata import CornellData
from chatbot.corpus.opensubsdata import OpensubsData
from chatbot.corpus.scotusdata import ScotusData
from chatbot.corpus.ubuntudata import UbuntuData
from chatbot.corpus.lightweightdata import LightweightData


class Batch:

    def __init__(self):
        self.encoderSeqs = []
        self.decoderSeqs = []
        self.targetSeqs = []
        self.weights = []


class TextData:

    availableCorpus = collections.OrderedDict([  # OrderedDict because the first element is the default choice
        ('cornell', CornellData),
        ('opensubs', OpensubsData),
        ('scotus', ScotusData),
        ('ubuntu', UbuntuData),
        ('lightweight', LightweightData),
    ])

    @staticmethod
    def corpusChoices():

        return list(TextData.availableCorpus.keys())

    def __init__(self, args):

        self.args = args

        self.corpusDir = os.path.join(self.args.rootDir, 'data', self.args.corpus)
        basePath = self._constructBasePath()
        self.fullSamplesPath = basePath + '.pkl'  # Full sentences length/vocab
        self.filteredSamplesPath = basePath + '-length{}-filter{}-vocabSize{}.pkl'.format(
            self.args.maxLength,
            self.args.filterVocab,
            self.args.vocabularySize,
        ) 

        self.padToken = -1  
        self.goToken = -1  
        self.eosToken = -1  
        self.unknownToken = -1  

        self.trainingSamples = []  

        self.word2id = {}
        self.id2word = {}  
        self.idCount = {}  
        self.loadCorpus()

        self._printStats()

        if self.args.playDataset:
            self.playDataset()

    def _printStats(self):
        print('Loaded {}: {} words, {} QA'.format(self.args.corpus, len(self.word2id), len(self.trainingSamples)))

    def _constructBasePath(self):
        path = os.path.join(self.args.rootDir, 'data' + os.sep + 'samples' + os.sep)
        path += 'dataset-{}'.format(self.args.corpus)
        if self.args.datasetTag:
            path += '-' + self.args.datasetTag
        return path

    def makeLighter(self, ratioDataset):
        pass



