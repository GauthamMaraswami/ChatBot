# ITA-Project-Chatbot
Creating a chatbot as a part of the ITA Course

# Project Team Members: -
Harshith Kumar 15CO120
Gautham M 15CO118 
Soham Ghosh 15MT35

# Project Name: Simple  Chabot
Language Intended to use Python 
Interface Command line (will try to provide browser Access If possible).

# Detailed Description:-
In this project we are trying to impliment simple chatbot using Ubuntu Corpus Dataset.This will be using Generative Model, Sequence To Sequence model introduced in Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation.Before training. We work on the dataset to convert the variable length sequences into fixed length sequences, by padding. We use a few special symbols to padd the sequence a. To Avoid too much padding that leads to extraneous computation,
Group sequences of similar lengths into the same buckets. And Create a separate subgraph for each bucket.
Block Diagram

![blockdiagramt](https://github.com/harshith47/ITA-Project-Chatbot/blob/master/seq2seq2.png)
How does the encoder work?
The encoder RNN conceives a sequence of context tokens one at a time and updates its hidden state. After processing the whole context sequence, it produces a final hidden state, which incorporates the sense of context and is used for generating the answer.
How does the decoder work?
The goal of the decoder is to take context representation from the encoder and generate an answer. For this purpose, a softmax layer over vocabulary is maintained in the decoder RNN. At each time
step, this layer takes the decoder hidden state and outputs a probability distribution over all words in its vocabulary.
Stages:
Stage 1: Learning the basic Requirements Like NLP and ML Algorithms
Stage 2: Implementing Naive bot with minimum functionality without using any algorithm
Stage 3: Using Algorithms to Improve the Speed, Expandability and Robustness of the code.
Stage 4: Expanding the Data set to make it more Interactive

# Work Distribution:
Stage                                 Gautham                                      Harshith                      Soham
    • 1                             Learning NLP                                 Learning ML                   Learning ML
    • 2                             Partial implications                     Partial implications       Partial Implications
    • 3                             NLP split algorithm                     Clustering Algorithms       NLP Split Algorithm
    • 4                             Test    Dataset                              Train Dataset               Test Dataset

# Input Output
Q:How re you? 
A : I am fine.
Q:What is the time ?
A:The time is 3:05
Q:Who are the people who make most of Highway Science in the Usa?
A:Prison Inmates

# Tools used
Python Jupyter
