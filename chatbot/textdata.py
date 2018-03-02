
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

    def shuffle(self):
        print('Shuffling the dataset...')
        random.shuffle(self.trainingSamples)

    def _createBatch(self, samples):

        batch = Batch()
        batchSize = len(samples)

        for i in range(batchSize):
            sample = samples[i]
            if not self.args.test and self.args.watsonMode:  
                sample = list(reversed(sample))
            if not self.args.test and self.args.autoEncode:  
                k = random.randint(0, 1)
                sample = (sample[k], sample[k])
             batch.encoderSeqs.append(list(reversed(sample[0])))  
            batch.decoderSeqs.append([self.goToken] + sample[1] + [self.eosToken]) 
            batch.targetSeqs.append(batch.decoderSeqs[-1][1:]) 
            assert len(batch.encoderSeqs[i]) <= self.args.maxLengthEnco
            assert len(batch.decoderSeqs[i]) <= self.args.maxLengthDeco
            batch.encoderSeqs[i]   = [self.padToken] * (self.args.maxLengthEnco  - len(batch.encoderSeqs[i])) + batch.encoderSeqs[i] 
            batch.weights.append([1.0] * len(batch.targetSeqs[i]) + [0.0] * (self.args.maxLengthDeco - len(batch.targetSeqs[i])))
            batch.decoderSeqs[i] = batch.decoderSeqs[i] + [self.padToken] * (self.args.maxLengthDeco - len(batch.decoderSeqs[i]))
            batch.targetSeqs[i]  = batch.targetSeqs[i]  + [self.padToken] * (self.args.maxLengthDeco - len(batch.targetSeqs[i]))

        encoderSeqsT = []  
        for i in range(self.args.maxLengthEnco):
            encoderSeqT = []
            for j in range(batchSize):
                encoderSeqT.append(batch.encoderSeqs[j][i])
            encoderSeqsT.append(encoderSeqT)
        batch.encoderSeqs = encoderSeqsT

        decoderSeqsT = []
        targetSeqsT = []
        weightsT = []
        for i in range(self.args.maxLengthDeco):
            decoderSeqT = []
            targetSeqT = []
            weightT = []
            for j in range(batchSize):
                decoderSeqT.append(batch.decoderSeqs[j][i])
                targetSeqT.append(batch.targetSeqs[j][i])
                weightT.append(batch.weights[j][i])
            decoderSeqsT.append(decoderSeqT)
            targetSeqsT.append(targetSeqT)
            weightsT.append(weightT)
        batch.decoderSeqs = decoderSeqsT
        batch.targetSeqs = targetSeqsT
        batch.weights = weightsT

        return batch

    def getBatches(self):
        self.shuffle()

        batches = []

        def genNextSamples():
 
            for i in range(0, self.getSampleSize(), self.args.batchSize):
                yield self.trainingSamples[i:min(i + self.args.batchSize, self.getSampleSize())]


        for samples in genNextSamples():
            batch = self._createBatch(samples)
            batches.append(batch)
        return batches

    def getSampleSize(self):
 
        return len(self.trainingSamples)

    def getVocabularySize(self):
       
        return len(self.word2id)

    def loadCorpus(self):
       
        datasetExist = os.path.isfile(self.filteredSamplesPath)
        if not datasetExist:  
            print('Training samples not found. Creating dataset...')

            datasetExist = os.path.isfile(self.fullSamplesPath)  
            if not datasetExist:
                print('Constructing full dataset...')

                optional = ''
                if self.args.corpus == 'lightweight':
                    if not self.args.datasetTag:
                        raise ValueError('Use the --datasetTag to define the lightweight file to use.')
                    optional = os.sep + self.args.datasetTag  
                corpusData = TextData.availableCorpus[self.args.corpus](self.corpusDir + optional)
                self.createFullCorpus(corpusData.getConversations())
                self.saveDataset(self.fullSamplesPath)
            else:
                self.loadDataset(self.fullSamplesPath)
            self._printStats()

            print('Filtering words (vocabSize = {} and wordCount > {})...'.format(
                self.args.vocabularySize,
                self.args.filterVocab
            ))
            self.filterFromFull()  
            print('Saving dataset...')
            self.saveDataset(self.filteredSamplesPath) 
        else:
            self.loadDataset(self.filteredSamplesPath)

        assert self.padToken == 0

    def saveDataset(self, filename):

        with open(os.path.join(filename), 'wb') as handle:
            data = {  
                'word2id': self.word2id,
                'id2word': self.id2word,
                'idCount': self.idCount,
                'trainingSamples': self.trainingSamples
            }
            pickle.dump(data, handle, -1)  

    def loadDataset(self, filename):

        dataset_path = os.path.join(filename)
        print('Loading dataset from {}'.format(dataset_path))
        with open(dataset_path, 'rb') as handle:
            data = pickle.load(handle)  
            self.word2id = data['word2id']
            self.id2word = data['id2word']
            self.idCount = data.get('idCount', None)
            self.trainingSamples = data['trainingSamples']

            self.padToken = self.word2id['<pad>']
            self.goToken = self.word2id['<go>']
            self.eosToken = self.word2id['<eos>']
            self.unknownToken = self.word2id['<unknown>']  


