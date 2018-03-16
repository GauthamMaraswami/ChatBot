
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

    def filterFromFull(self):

        def mergeSentences(sentences, fromEnd=False):
           
            merged = []

            if fromEnd:
                sentences = reversed(sentences)

            for sentence in sentences:

                if len(merged) + len(sentence) <= self.args.maxLength:
                    if fromEnd: 
                        merged = sentence + merged
                    else:
                        merged = merged + sentence
                else:  
                    for w in sentence:
                        self.idCount[w] -= 1
            return merged

        newSamples = []

        for inputWords, targetWords in tqdm(self.trainingSamples, desc='Filter sentences:', leave=False):
            inputWords = mergeSentences(inputWords, fromEnd=True)
            targetWords = mergeSentences(targetWords, fromEnd=False)

            newSamples.append([inputWords, targetWords])
        words = []

        specialTokens = { 
            self.padToken,
            self.goToken,
            self.eosToken,
            self.unknownToken
        }
        newMapping = {}  
        newId = 0

        selectedWordIds = collections \
            .Counter(self.idCount) \
            .most_common(self.args.vocabularySize or None)  
        selectedWordIds = {k for k, v in selectedWordIds if v > self.args.filterVocab}
        selectedWordIds |= specialTokens

        for wordId, count in [(i, self.idCount[i]) for i in range(len(self.idCount))]:  
            if wordId in selectedWordIds:  # Update the word id
                newMapping[wordId] = newId
                word = self.id2word[wordId]  
                del self.id2word[wordId]  
                self.word2id[word] = newId
                self.id2word[newId] = word
                newId += 1
            else: 
                newMapping[wordId] = self.unknownToken
                del self.word2id[self.id2word[wordId]]  # The word isn't used anymore
                del self.id2word[wordId]

        def replace_words(words):
            valid = False  # Filter empty sequences
            for i, w in enumerate(words):
                words[i] = newMapping[w]
                if words[i] != self.unknownToken:  
                    valid = True
            return valid

        self.trainingSamples.clear()

        for inputWords, targetWords in tqdm(newSamples, desc='Replace ids:', leave=False):
            valid = True
            valid &= replace_words(inputWords)
            valid &= replace_words(targetWords)
            valid &= targetWords.count(self.unknownToken) == 0  

            if valid:
                self.trainingSamples.append([inputWords, targetWords]) 

        self.idCount.clear()  

    def createFullCorpus(self, conversations):

        self.padToken = self.getWordId('<pad>')  
        self.goToken = self.getWordId('<go>') 
        self.eosToken = self.getWordId('<eos>')  
        self.unknownToken = self.getWordId('<unknown>') 


        for conversation in tqdm(conversations, desc='Extract conversations'):
            self.extractConversation(conversation)

    def extractConversation(self, conversation):
        if self.args.skipLines: 
            step = 2
        else:
            step = 1

        for i in tqdm_wrap(
            range(0, len(conversation['lines']) - 1, step),  # We ignore the last line (no answer for it)
            desc='Conversation',
            leave=False
        ):
            inputLine  = conversation['lines'][i]
            targetLine = conversation['lines'][i+1]

            inputWords  = self.extractText(inputLine['text'])
            targetWords = self.extractText(targetLine['text'])

            if inputWords and targetWords:  
                self.trainingSamples.append([inputWords, targetWords])

    def extractText(self, line):

        sentences = [] 
        sentencesToken = nltk.sent_tokenize(line)

        for i in range(len(sentencesToken)):
            tokens = nltk.word_tokenize(sentencesToken[i])

            tempWords = []
            for token in tokens:
                tempWords.append(self.getWordId(token)) 

            sentences.append(tempWords)

        return sentences

    def getWordId(self, word, create=True):

        word = word.lower()  
        if not create:
            wordId = self.word2id.get(word, self.unknownToken)
        elif word in self.word2id:
            wordId = self.word2id[word]
            self.idCount[wordId] += 1
        else:
            wordId = len(self.word2id)
            self.word2id[word] = wordId
            self.id2word[wordId] = word
            self.idCount[wordId] = 1

        return wordId

    def printBatch(self, batch):

        print('----- Print batch -----')
        for i in range(len(batch.encoderSeqs[0])):  # Batch size
            print('Encoder: {}'.format(self.batchSeq2str(batch.encoderSeqs, seqId=i)))
            print('Decoder: {}'.format(self.batchSeq2str(batch.decoderSeqs, seqId=i)))
            print('Targets: {}'.format(self.batchSeq2str(batch.targetSeqs, seqId=i)))
            print('Weights: {}'.format(' '.join([str(weight) for weight in [batchWeight[i] for batchWeight in batch.weights]])))



