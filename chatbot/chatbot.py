import argparse  
import configparser  
import datetime  
import os  
import tensorflow as tf
import numpy as np
import math
from tqdm import tqdm  
from tensorflow.python import debug as tf_debug
from chatbot.textdata import TextData
from chatbot.model import Model
class Chatbot:
    class TestMode:
        ALL = 'all'
        INTERACTIVE = 'interactive' 

    def __init__(self):
        self.args = None
        self.textData = None  
        self.model = None  
        self.writer = None
        self.saver = None
        self.modelDir = ''  
        self.globStep = 0 
        self.sess = None
        self.MODEL_DIR_BASE = 'save' + os.sep + 'model'
        self.MODEL_NAME_BASE = 'model'
        self.MODEL_EXT = '.ckpt'
        self.CONFIG_FILENAME = 'params.ini'
        self.CONFIG_VERSION = '0.5'
        self.TEST_IN_NAME = 'data' + os.sep + 'test' + os.sep + 'samples.txt'
        self.TEST_OUT_SUFFIX = '_predictions.txt'
        self.SENTENCES_PREFIX = ['Q: ', 'A: ']

    @staticmethod
    def parseArgs(args):
        parser = argparse.ArgumentParser()
        globalArgs = parser.add_argument_group('Global options')
        globalArgs.add_argument('--test',
                                nargs='?',
                                choices=[Chatbot.TestMode.ALL, Chatbot.TestMode.INTERACTIVE, Chatbot.TestMode.DAEMON],
                                const=Chatbot.TestMode.ALL, default=None,
                                help='if ')
        

        nnArgs = parser.add_argument_group('Network options', 'architecture related option')
        trainingArgs = parser.add_argument_group('Training options')
        return parser.parse_args(args)

    def main(self, args=None):
        print('TensorFlow detected: v{}'.format(tf.__version__))
        self.args = self.parseArgs(args)
        if not self.args.rootDir:
            self.args.rootDir = os.getcwd()  
        self.loadModelParams()  
        self.textData = TextData(self.args)
        if self.args.createDataset:
            
        with tf.device(self.getDevice()):
            self.model = Model(self.args, self.textData)

        self.writer = tf.summary.FileWriter(self._getSummaryName())
        self.saver = tf.train.Saver(max_to_keep=200)

     
        self.sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True, 
            log_device_placement=False)  
        ) 

        if self.args.debug:
            self.sess = tf_debug.LocalCLIDebugWrapperSession(self.sess)
            self.sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

        print('Initialize variables...')
        self.sess.run(tf.global_variables_initializer())

        if self.args.test != Chatbot.TestMode.ALL:
            self.managePreviousModel(self.sess)

        if self.args.initEmbeddings:
            self.loadEmbedding(self.sess)

        if self.args.test:
            if self.args.test == Chatbot.TestMode.INTERACTIVE:
                self.mainTestInteractive(self.sess)
            elif self.args.test == Chatbot.TestMode.ALL:
                print('Start predicting...')
                self.predictTestset(self.sess)
                print('All predictions done')
            elif self.args.test == Chatbot.TestMode.DAEMON:
                print('Daemon mode, running in background...')
            else:
                raise RuntimeError('Unknown test mode: {}'.format(self.args.test)) 
        else:
            self.mainTrain(self.sess)

        if self.args.test != Chatbot.TestMode.DAEMON:
            self.sess.close()
            print("The End! Thanks for using this program")

    def mainTrain(self, sess):

        self.textData.makeLighter(self.args.ratioDataset) 

        mergedSummaries = tf.summary.merge_all()  
        if self.globStep == 0:  
            self.writer.add_graph(sess.graph)  


        print('Start training (press Ctrl+C to save and exit)...')

        try: 
            for e in range(self.args.numEpochs):

                print()
                print("----- Epoch {}/{} ; (lr={}) -----".format(e+1, self.args.numEpochs, self.args.learningRate))

                batches = self.textData.getBatches()

                tic = datetime.datetime.now()
                for nextBatch in tqdm(batches, desc="Training"):
                    ops, feedDict = self.model.step(nextBatch)
                    assert len(ops) == 2  
                    _, loss, summary = sess.run(ops + (mergedSummaries,), feedDict)
                    self.writer.add_summary(summary, self.globStep)
                    self.globStep += 1

                    if self.globStep % 100 == 0:
                        perplexity = math.exp(float(loss)) if loss < 300 else float("inf")
                        tqdm.write("----- Step %d -- Loss %.2f -- Perplexity %.2f" % (self.globStep, loss, perplexity))

                    if self.globStep % self.args.saveEvery == 0:
                        self._saveSession(sess)

                toc = datetime.datetime.now()

                print("Epoch finished in {}".format(toc-tic))  
        except (KeyboardInterrupt, SystemExit):  
            print('Interruption detected, exiting the program...')

        self._saveSession(sess) 

    def predictTestset(self, sess):

        with open(os.path.join(self.args.rootDir, self.TEST_IN_NAME), 'r') as f:
            lines = f.readlines()

        modelList = self._getModelList()
        if not modelList:
            print('Warning: No model found in \'{}\'. Please train a model before trying to predict'.format(self.modelDir))
            return

        for modelName in sorted(modelList):  
            print('Restoring previous model from {}'.format(modelName))
            self.saver.restore(sess, modelName)
            print('Testing...')

            saveName = modelName[:-len(self.MODEL_EXT)] + self.TEST_OUT_SUFFIX 
            with open(saveName, 'w') as f:
                nbIgnored = 0
                for line in tqdm(lines, desc='Sentences'):
                    question = line[:-1]  

                    answer = self.singlePredict(question)
                    if not answer:
                        nbIgnored += 1
                        continue  

                    predString = '{x[0]}{0}\n{x[1]}{1}\n\n'.format(question, self.textData.sequence2str(answer, clean=True), x=self.SENTENCES_PREFIX)
                    if self.args.verbose:
                        tqdm.write(predString)
                    f.write(predString)
                print('Prediction finished, {}/{} sentences ignored (too long)'.format(nbIgnored, len(lines)))

    def mainTestInteractive(self, sess):

        
        while True:
            question = input(self.SENTENCES_PREFIX[0])
            if question == '' or question == 'exit':
                break

            questionSeq = []  
            answer = self.singlePredict(question, questionSeq)
            if not answer:
                print('Warning: sentence too long, sorry. Maybe try a simpler sentence.')
                continue 

            print('{}{}'.format(self.SENTENCES_PREFIX[1], self.textData.sequence2str(answer, clean=True)))

            if self.args.verbose:
                print(self.textData.batchSeq2str(questionSeq, clean=True, reverse=True))
                print(self.textData.sequence2str(answer))

            print()

    def singlePredict(self, question, questionSeq=None):
        batch = self.textData.sentence2enco(question)
        if not batch:
            return None
        if questionSeq is not None:  
            questionSeq.extend(batch.encoderSeqs)

        ops, feedDict = self.model.step(batch)
        output = self.sess.run(ops[0], feedDict) 
        answer = self.textData.deco2sentence(output)

        return answer

    def daemonPredict(self, sentence):
        return self.textData.sequence2str(
            self.singlePredict(sentence),
            clean=True
        )

    def daemonClose(self):

        print('Exiting the daemon mode...')
        self.sess.close()
        print('Daemon closed.')

    def loadEmbedding(self, sess):

        with tf.variable_scope("embedding_rnn_seq2seq/rnn/embedding_wrapper", reuse=True):
            em_in = tf.get_variable("embedding")
        with tf.variable_scope("embedding_rnn_seq2seq/embedding_rnn_decoder", reuse=True):
            em_out = tf.get_variable("embedding")

        variables = tf.get_collection_ref(tf.GraphKeys.TRAINABLE_VARIABLES)
        variables.remove(em_in)
        variables.remove(em_out)

        if self.globStep != 0:
            return

        embeddings_path = os.path.join(self.args.rootDir, 'data', 'embeddings', self.args.embeddingSource)
        embeddings_format = os.path.splitext(embeddings_path)[1][1:]
        print("Loading pre-trained word embeddings from %s " % embeddings_path)
        with open(embeddings_path, "rb") as f:
            header = f.readline()
            vocab_size, vector_size = map(int, header.split())
            binary_len = np.dtype('float32').itemsize * vector_size
            initW = np.random.uniform(-0.25,0.25,(len(self.textData.word2id), vector_size))
            for line in tqdm(range(vocab_size)):
                word = []
                while True:
                    ch = f.read(1)
                    if ch == b' ':
                        word = b''.join(word).decode('utf-8')
                        break
                    if ch != b'\n':
                        word.append(ch)
                if word in self.textData.word2id:
                    if embeddings_format == 'bin':
                        vector = np.fromstring(f.read(binary_len), dtype='float32')
                    elif embeddings_format == 'vec':
                        vector = np.fromstring(f.readline(), sep=' ', dtype='float32')
                    else:
                        raise Exception("Unkown format for embeddings: %s " % embeddings_format)
                    initW[self.textData.word2id[word]] = vector
                else:
                    if embeddings_format == 'bin':
                        f.read(binary_len)
                    elif embeddings_format == 'vec':
                        f.readline()
                    else:
                        raise Exception("Unkown format for embeddings: %s " % embeddings_format)

        if self.args.embeddingSize < vector_size:
            U, s, Vt = np.linalg.svd(initW, full_matrices=False)
            S = np.zeros((vector_size, vector_size), dtype=complex)
            S[:vector_size, :vector_size] = np.diag(s)
            initW = np.dot(U[:, :self.args.embeddingSize], S[:self.args.embeddingSize, :self.args.embeddingSize])

        sess.run(em_in.assign(initW))
        sess.run(em_out.assign(initW))


    def managePreviousModel(self, sess):

        print('WARNING: ', end='')

        modelName = self._getModelName()

        if os.listdir(self.modelDir):
            if self.args.reset:
                print('Reset: Destroying previous model at {}'.format(self.modelDir))
            elif os.path.exists(modelName): 
                print('Restoring previous model from {}'.format(modelName))
                self.saver.restore(sess, modelName)  
            elif self._getModelList():
                print('Conflict with previous models.')
                raise RuntimeError('Some models are already present in \'{}\'. You should check them first (or re-try with the keepAll flag)'.format(self.modelDir))
            else: 
                print('No previous model found, but some files found at {}. Cleaning...'.format(self.modelDir)) 
                self.args.reset = True

            if self.args.reset:
                fileList = [os.path.join(self.modelDir, f) for f in os.listdir(self.modelDir)]
                for f in fileList:
                    print('Removing {}'.format(f))
                    os.remove(f)
        else:
            print('No previous model found, starting from clean directory: {}'.format(self.modelDir))

    def _saveSession(self, sess):

        tqdm.write('Checkpoint reached: saving model (don\'t stop the run)...')
        self.saveModelParams()
        model_name = self._getModelName()
        with open(model_name, 'w') as f:  
            f.write('This file is used internally by DeepQA to check the model existance. Please do not remove.\n')
        self.saver.save(sess, model_name) 
        tqdm.write('Model saved.')

    def _getModelList(self):
        return [os.path.join(self.modelDir, f) for f in os.listdir(self.modelDir) if f.endswith(self.MODEL_EXT)]

    def loadModelParams(self):

        self.modelDir = os.path.join(self.args.rootDir, self.MODEL_DIR_BASE)
        if self.args.modelTag:
            self.modelDir += '-' + self.args.modelTag

        configName = os.path.join(self.modelDir, self.CONFIG_FILENAME)
        if not self.args.reset and not self.args.createDataset and os.path.exists(configName):

            config = configparser.ConfigParser()
            config.read(configName)

            currentVersion = config['General'].get('version')
            if currentVersion != self.CONFIG_VERSION:
                raise UserWarning('Present configuration version {0} does not match {1}. You can try manual changes on \'{2}\''.format(currentVersion, self.CONFIG_VERSION, configName))

            self.globStep = config['General'].getint('globStep')
            self.args.watsonMode = config['General'].getboolean('watsonMode')
            self.args.autoEncode = config['General'].getboolean('autoEncode')
            self.args.corpus = config['General'].get('corpus')

            self.args.datasetTag = config['Dataset'].get('datasetTag')
            self.args.maxLength = config['Dataset'].getint('maxLength')  
            self.args.filterVocab = config['Dataset'].getint('filterVocab')
            self.args.skipLines = config['Dataset'].getboolean('skipLines')
            self.args.vocabularySize = config['Dataset'].getint('vocabularySize')

            self.args.hiddenSize = config['Network'].getint('hiddenSize')
            self.args.numLayers = config['Network'].getint('numLayers')
            self.args.softmaxSamples = config['Network'].getint('softmaxSamples')
            self.args.initEmbeddings = config['Network'].getboolean('initEmbeddings')
            self.args.embeddingSize = config['Network'].getint('embeddingSize')
            self.args.embeddingSource = config['Network'].get('embeddingSource')

            print()
            print('Warning: Restoring parameters:')
            print('globStep: {}'.format(self.globStep))
            print('watsonMode: {}'.format(self.args.watsonMode))
            print('autoEncode: {}'.format(self.args.autoEncode))
            print('corpus: {}'.format(self.args.corpus))
            print('datasetTag: {}'.format(self.args.datasetTag))
            print('maxLength: {}'.format(self.args.maxLength))
            print('filterVocab: {}'.format(self.args.filterVocab))
            print('skipLines: {}'.format(self.args.skipLines))
            print('vocabularySize: {}'.format(self.args.vocabularySize))
            print('hiddenSize: {}'.format(self.args.hiddenSize))
            print('numLayers: {}'.format(self.args.numLayers))
            print('softmaxSamples: {}'.format(self.args.softmaxSamples))
            print('initEmbeddings: {}'.format(self.args.initEmbeddings))
            print('embeddingSize: {}'.format(self.args.embeddingSize))
            print('embeddingSource: {}'.format(self.args.embeddingSource))
            print()

        self.args.maxLengthEnco = self.args.maxLength
        self.args.maxLengthDeco = self.args.maxLength + 2

        if self.args.watsonMode:
            self.SENTENCES_PREFIX.reverse()


    def saveModelParams(self):
        config = configparser.ConfigParser()
        config['General'] = {}
        config['General']['version']  = self.CONFIG_VERSION
        config['General']['globStep']  = str(self.globStep)
        config['General']['watsonMode'] = str(self.args.watsonMode)
        config['General']['autoEncode'] = str(self.args.autoEncode)
        config['General']['corpus'] = str(self.args.corpus)

        config['Dataset'] = {}
        config['Dataset']['datasetTag'] = str(self.args.datasetTag)
        config['Dataset']['maxLength'] = str(self.args.maxLength)
        config['Dataset']['filterVocab'] = str(self.args.filterVocab)
        config['Dataset']['skipLines'] = str(self.args.skipLines)
        config['Dataset']['vocabularySize'] = str(self.args.vocabularySize)

        config['Network'] = {}
        config['Network']['hiddenSize'] = str(self.args.hiddenSize)
        config['Network']['numLayers'] = str(self.args.numLayers)
        config['Network']['softmaxSamples'] = str(self.args.softmaxSamples)
        config['Network']['initEmbeddings'] = str(self.args.initEmbeddings)
        config['Network']['embeddingSize'] = str(self.args.embeddingSize)
        config['Network']['embeddingSource'] = str(self.args.embeddingSource)

        # Keep track of the learning params (but without restoring them)
        config['Training (won\'t be restored)'] = {}
        config['Training (won\'t be restored)']['learningRate'] = str(self.args.learningRate)
        config['Training (won\'t be restored)']['batchSize'] = str(self.args.batchSize)
        config['Training (won\'t be restored)']['dropout'] = str(self.args.dropout)

        with open(os.path.join(self.modelDir, self.CONFIG_FILENAME), 'w') as configFile:
            config.write(configFile)

    def _getSummaryName(self):

        return self.modelDir

    def _getModelName(self):

        modelName = os.path.join(self.modelDir, self.MODEL_NAME_BASE)
        if self.args.keepAll:
            modelName += '-' + str(self.globStep)
        return modelName + self.MODEL_EXT

    def getDevice(self):

        if self.args.device == 'cpu':
            return '/cpu:0'
        elif self.args.device == 'gpu':
            return '/gpu:0'
        elif self.args.device is None:  
            return None
        else:
            print('Warning: Error in the device name: {}, use the default device'.format(self.args.device))
            return None
