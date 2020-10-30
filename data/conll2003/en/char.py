#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#DISCLAMER: the code above uses a lot of existing projects and explanatory posts including batching part made after: http://digtime.cn/articles/222/ai-for-trading-character-level-lstm-in-pytorch-98 
#
#


import torch
from torch import nn
import time
import torchtext
import numpy as np

import torch.nn as nn
from torch.nn import Module, Embedding, LSTM, Linear, NLLLoss, Dropout, CrossEntropyLoss
import torch.optim as optim
import random
import torch.nn.functional as F
import torch.optim as optim
import nltk



from collections import defaultdict, Counter

import matplotlib.pyplot as plt
from pathlib import Path

from random import choice, random, shuffle
import sys

#from model import Model
from numpy import argmax




pathtodirectory = Path("C:\\Users\\LubaC\\Desktop\\BILSTM\\scripts\\data\\conll2003\\en")
corpus_file = pathtodirectory / "train.txt"

# define input string
data = corpus_file
#print(data)
# define universe of possible input values
alphabet = 'abcdefghijklmnopqrstuvwxyz '
# define a mapping of chars to integers
char_to_int = dict((c, i) for i, c in enumerate(alphabet))
int_to_char = dict((i, c) for i, c in enumerate(alphabet))
# integer encode input data
integer_encoded = [char_to_int[char] for char in data]
print(integer_encoded)
# one hot encode
onehot_encoded = list()
for value in integer_encoded:
	letter = [0 for _ in range(len(alphabet))]
	letter[value] = 1
	onehot_encoded.append(letter)
print(onehot_encoded)
# invert encoding
inverted = int_to_char[argmax(onehot_encoded[0])]
print(inverted)

def generate_batches(data, batch_size, shuffle=True, drop_last=True, device="cpu"):

 """ A generator function which wraps the PyTorch DataLoader. It will ensure each tensor is on the write device location. """ 
 
	dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last) 
	for data_dict in dataloader: 
		out_data_dict = {} 
		for name, tensor in data_dict.items(): 
			out_data_dict[name] = data_dict[name].to(device) 
		yield out_data_dict
		
		
		
		
class CharLSTM(nn.Module):
    """ A Classifier with an RNN to extract features and an MLP to classify """
    def __init__(self, embedding_size, num_embeddings, num_classes,
                 rnn_hidden_size, batch_first=True, padding_idx=0):
        """
        Args:
            embedding_size (int): The size of the character embeddings
            num_embeddings (int): The number of characters to embed
            num_classes (int): The size of the prediction vector 
                Note: the number of nationalities
            rnn_hidden_size (int): The size of the RNN's hidden state
            batch_first (bool): Informs whether the input tensors will 
                have batch or the sequence on the 0th dimension
            padding_idx (int): The index for the tensor padding; 
                see torch.nn.Embedding
        """
        super (CharLSTM, self).__init__()

        self.emb = nn.Embedding(num_embeddings=num_embeddings,
                                embedding_dim=embedding_size,
                                padding_idx=padding_idx)
        self.rnn = ElmanRNN(input_size=embedding_size,
                             hidden_size=rnn_hidden_size,
                             batch_first=batch_first)
        self.fc1 = nn.Linear(in_features=rnn_hidden_size,
                         out_features=rnn_hidden_size)
        self.fc2 = nn.Linear(in_features=rnn_hidden_size,
                          out_features=num_classes)

    def forward(self, x_in, x_lengths=None, apply_softmax=False):
        """The forward pass of the classifier
        
        Args:
            x_in (torch.Tensor): an input data tensor. 
                x_in.shape should be (batch, input_dim)
            x_lengths (torch.Tensor): the lengths of each sequence in the batch.
                They are used to find the final vector of each sequence
            apply_softmax (bool): a flag for the softmax activation
                should be false if used with the Cross Entropy losses
        Returns:
            the resulting tensor. tensor.shape should be (batch, output_dim)
        """
        x_embedded = self.emb(x_in)
        y_out = self.rnn(x_embedded)

        if x_lengths is not None:
            y_out = column_gather(y_out, x_lengths)
        else:
            y_out = y_out[:, -1, :]

        y_out = F.relu(self.fc1(F.dropout(y_out, 0.5)))
        y_out = self.fc2(F.dropout(y_out, 0.5))

        if apply_softmax:
            y_out = F.softmax(y_out, dim=1)

        return y_out
		
# create the network after predefined module
net = CharLSTM(sequence_len=128, vocab_size=len(char2int), hidden_dim=50, batch_size=128)
#charmodel = CharLSTM(pretrained_embeds, 100, len(word_to_idx), n_classes)
# loss and the optimizer like in main lstm
loss_function = NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.6)


# Training loop
    for e in range(n_epochs+1):
        total_loss = 0
        for sent in data["train"][5:10]:
            
            # (1) Set gradient to zero for new example: Set gradients to zero before pass
            model.zero_grad()
            
            # (2) Encode sentence and tag sequence as sequences of indices
            input_sent,  gold_tags = prepare_emb(sent["TOKENS"], sent["NE"], word_to_idx, TAG_INDICES)
            
            # (3) Predict tags (sentence by sentence)
            if len(input_sent) > 0:
                pred_scores = model(input_sent)
                
                # (4) Compute loss and do backward step
                loss = loss_function(pred_scores, gold_tags)
                loss.backward()
              
                # (5) Optimize parameter values
                optimizer.step()
          
                # (6) Accumulate loss
                total_loss += loss
        if ((e+1) % report_every) == 0:
            print('epoch: %d, loss: %.4f' % (e, total_loss*100/len(data['train'])))
            
    # Save the trained model
    save_model(model, pathtodirectory+ 'm2.pkl')
    
    # Load model from file
    load_model(pathtodirectory + 'm2.pkl')
	
	
	
correct = 0
with torch.no_grad():
  for sent in data["test"]:
    input_sent,  gold_tags = prepare_emb(sent["TOKENS"], sent["NE"], word_to_idx, TAG_INDICES)
    # WRITE CODE HERE
    predicted, correct = 0.0, 0.0
    
    # Predict class with the highest probability
    if len(input_sent) > 0:
        predicted = torch.argmax(model(input_sent), dim=1)
        print(predicted)
        print(gold_tags)
        correct += torch.eq(predicted,gold_tags).item()
  
    if verbose:
      print('TEST DATA: %s, OUTPUT: %s, GOLD TAG: %d' % 
            (sent["TOKENS"], sent["NE"], predicted))
      
  print('test accuracy: %.2f' % (100.0 * correct / len(data["test"])))
#  print('test recall: %.2f' % (100.0 * correct / len(data["test"]))) 
 # print('test f1: %.2f' % (100.0 * correct / len(data["test"])))
  
#print('F1: {}'.format(f1_score(outGT, outPRED, average="samples")))
#print('Precision: {}'.format(precision_score(outGT, outPRED, average="samples")))
#print('Recall: {}'.format(recall_score(outGT, outPRED, average="samples")))

		
		
"""		
class CharRNN(nn.Module):

    def __init__(self, tokens, n_hidden=256, n_layers=2,
                               drop_prob=0.5, lr=0.001):
        super().__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.lr = lr

        # creating character dictionaries
        self.chars = tokens
        self.int2char = dict(enumerate(self.chars))
        self.char2int = {ch: ii for ii, ch in self.int2char.items()}

        ## TODO: define the layers of the model
        self.lstm = nn.LSTM(len(self.chars), n_hidden, n_layers,
                           dropout=drop_prob, batch_first=True)

        ## TODO:: define a dropout layer
        self.dropout = nn.Dropout(drop_prob)

        ## TODO: define the final, fully-connected output layer
        self.fc = nn.Linear(n_hidden, len(self.chars))

    def forward(self, x, hidden):
        ''' Forward pass through the network. 
            These inputs are x, and the hidden/cell state `hidden`. '''

        ## TODO: Get the outputs and the new hidden state from the lstm
        r_output, hidden = self.lstm(x, hidden)

        ## TODO::pass through a dropout layer
        out = self.dropout(r_output)

        # Stack up LSTM outputs using view
        # you may need to use contiguous to reshape the output
        out = out.contiguous().view(-1, self.n_hidden)

         ## TODO: put x through the fully-connected layer
        out = self.fc(out)

        # return the final output and the hidden state
        return out, hidden

    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data

        if (train_on_gpu):
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
                  weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_())

        return hidden

"""


