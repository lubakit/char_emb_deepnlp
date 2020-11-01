
"""
DISCLAMER: the code above uses assignment 3 parts, examples from existing projects and explanatory posts, mainly not taking te code directly, but builsing the same structures
including tokenization and pre-processing : http://digtime.cn/articles/222/ai-for-trading-character-level-lstm-in-pytorch-98
character embeddings based on word embeddings from pytorch website, https://towardsdatascience.com/implementing-word2vec-in-pytorch-skip-gram-model-e6bae040d2fb

Some parts made by the project partner Karina Hensel were changed and imported as functions or taken and changed for more compatibility for the planned but not zet realised concatenating
As it was not possible to test the code only the building of the embeddings(not realised after because of no technical possibiity) and original code is changed back to original slightly modified Karinas code
"""



import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import codecs
import math


import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Embedding, LSTM, Linear, NLLLoss, Dropout, CrossEntropyLoss
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchtext
from torchtext import data

import gensim
from gensim.scripts.glove2word2vec import glove2word2vec

import numpy as np
import os
import math
from utils import *

import numpy as np
import os
import math
from model.utils import *
from model.model import Model

# Data and hyperparameters in case custom-made embeddings would not work

#File we have our embeddings from. They are pretrained and taken directly from character pretrained embeddings https://github.com/minimaxir/char-embeddings/blob/master/glove.840B.300d-char.txt (see file charemb.txt)
embeddings_file = '../data/embeddings/en/charemb.txt'
data_dir = '../data/conll2003/en/'
model_dir = '../models/'

#from word enmbeddings code

def prepare_emb(sent, tags, chars_to_ix, tags_to_ix):
    ch_idxs, tag_idxs = [], []
    for w, t in zip(sent, tags):
        if w.lower() in chars_to_ix.keys():
            w_idxs.append(chars_to_ix[w.lower()])  # = [to_ix[w] for w in seq if w in to_ix.keys()]
        else:
            # Use 'Frock' as dummy for unknown words (only temporary solution)
            ch_idxs.append(chars_to_ix['frock'])

        if t in tags_to_ix.keys():
            tag_idxs.append(tags_to_ix[t])
        else:
            tag_idxs.append(tags_to_ix['O'])

    return torch.tensor(ch_idxs, dtype=torch.long), torch.tensor(tag_idxs, dtype=torch.long)



"""does not work. From here no pretrained character embeddings are used. Path to the file with sentences from the corpus without labels. The corpus we create the model for is on the same folder"""
pathtodirectory = Path("C:\\Users\\LubaC\\Desktop\\character_emb\\scripts\\data\\conll2003\\en")
filepath = pathtodirectory / "words.txt"


with open(filepath, 'r', encoding='utf-8') as f:
    text = f.read()

#data = load_sentences(filepath)
#print (data)
#sent = sentence['TOKENS']
#print(sent)

#testing how is imported: first 200 characters
#print(text[:200])
"""Tokenization of the text before creating of the character embeddings"""

#importing characters
chars = tuple(set(text))
#int2char dictionary maps integers to characters
int2char = dict(enumerate(chars))
#dictionary maps characters to their indexes
char2int = {ch: ii for ii, ch in int2char.items()}
# encode the text
encoded = np.array([char2int[ch] for ch in text])

#testing how imported
#print(text[:200])
#prints all characters
print (chars)
print (len(chars))
#prints the mapping of integers to characters
#print(int2char)
#prints the mapping of characters to integers
#print (char2int)
#print (encoded[:])

"""Preproceccing the data: building the one-hot-encoding that will be the input to the LSTM to create the character embeddings"""
def one_hot_encoding(arr, n_labels):
    # Initialize an empty array
    one_hot = np.zeros((arr.size, n_labels), dtype=np.float32)

    # Filling 1 where needed to create uniwue representations
    one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1.

    # reshaping to original
    one_hot = one_hot.reshape((*arr.shape, n_labels))

    return one_hot

#testing
#n_chars = 85
#x = one_hot_encoding(encoded, n_chars)
#inputs, labels = torch.from_numpy(x), torch.from_numpy(x)
#print (x)



"""As we cannot have the one hot enodings as the imput (they do not contain semantic connection information)
 sequence of one hot encodings can turn into an embedding after giving it into the LSTM or nn.Embedding 
 One hot encodding is made into the embedding  by squashing the dimensional space
 In our case similar vectors to similar spelling of characters
 Character 1 and character 2 are both independently given to the LSTM, their norm of differences is calculated and given to the sigmoid function. 
 In the end we have the similar spelled characetrs represented via similarly "spelled" vectors"""
#def one_hot_to_embedding(batch,depth):
#    emb = nn.Embedding(depth, depth)
#    emb.weight.data = torch.eye(depth)
#    return emb(batch)
#fron pytorch website,  how to create embeddings
#word_to_ix = {"hello": 0, "world": 1}
#embeds = nn.Embedding(85, 5)  # 85 chars in vocab, 5 dimensional embeddings
#lookup_tensor = torch.tensor([word_to_ix["hello"]], dtype=torch.long)
#hello_embed = embeds(lookup_tensor)
#print(hello_embed)





#Character LSTM
class LSTMModel(nn.Module):
    def __init__(self, pretrained_embeddings, hidden_size, vocab_size, n_classes):
        
        super(Model, self).__init__()
        
        # Vocabulary size
        self.vocab_size = pretrained_embeddings.shape[0]
        # Embedding dimensionality
        self.embedding_size = pretrained_embeddings.shape[1]
        # Number of hidden units
        self.hidden_size = hidden_size
        
        # Embedding layer
        self.embedding = Embedding(self.vocab_size, self.embedding_size)
        
        # Dropout
        self.dropout = Dropout(p=0.5, inplace=False)
        # Hidden layer (300, 20)
        self.lstm = LSTM(self.embedding_size, self.hidden_size, num_layers=2)
        # Final prediction layer
        self.hidden2tag = Linear(self.hidden_size, n_classes)#, bias=True)
    
    def forward(self, x):
        # Retrieve word embedding for input token
        emb = self.embedding(x)
        # Apply dropout
        dropout = self.dropout(emb)
        # Hidden layer
        h, _ = self.lstm(emb.view(len(x), 1, -1))
        # Prediction
        pred = self.hidden2tag(h.view(len(x), -1))
        
        return F.log_softmax(pred, dim=1)

#Hyperparameteres and training

embeddings_file = '../data/embeddings/en/charemb.txt'
data_dir = '../data/conll2003/en/'
model_dir = '../model/'
model = 'model_char.pkl'

if __name__=='__main__':
    # Load data
    data = read_conll_datasets(data_dir)
    gensim_embeds = gensim.models.KeyedVectors.load_word2vec_format(embeddings_file, encoding='utf8')
    pretrained_embeds = gensim_embeds.vectors 
    
    # To convert words in the input to indices of the embeddings matrix:
    char_to_idx = {char: i for i, word in enumerate(gensim_embeds.vocab.keys())}
    
    # Hyperparameters
    # Number of output classes (9)
    n_classes = len(TAG_INDICES)
    # Epochs
    n_epochs = 1
    # Batch size (currently not used)
    batch_size = 32
    report_every = 1
    verbose = True
    
    # Set up and initialize model
    model = Model(pretrained_embeds, 100, len(char_to_idx), n_classes)
    loss_function = NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.6)
    
    # Training loop
    for e in range(n_epochs+1):
        total_loss = 0
        for sent in data["train"][5:10]:
            
            # (1) Set gradient to zero for new example
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
    save_model(model, 'model_char.pkl')

running_loss += loss.item()
        if i % 1000 == 999:    # every 1000 mini-batches...

            # ...log the running loss
            writer.add_scalar('training loss',
                            running_loss / 1000,
                            epoch * len(data['train'] + i)


data_dir = '../data/conll2003/en/'
model_dir = '../model/'
model = 'm1.pkl'

if __name__=='__main__':
    # Load data
    data = read_conll_datasets(data_dir)
    gensim_embeds = gensim.models.KeyedVectors.load_word2vec_format(embeddings_file, encoding='utf8')
    pretrained_embeds = gensim_embeds.vectors 
    
    # To convert words in the input to indices of the embeddings matrix:
    word_to_idx = {word: i for i, word in enumerate(gensim_embeds.vocab.keys())}
    
    # Hyperparameters
    # Number of output classes (9)
    n_classes = len(TAG_INDICES)
    # Epochs
    n_epochs = 1
    # Batch size (currently not used)
    batch_size = 32
    report_every = 1
    verbose = True
    
    # Set up and initialize model
    model = Model(pretrained_embeds, 100, len(word_to_idx), n_classes)
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
    save_model(model, 'm1.pkl')

running_loss += loss.item()
        if i % 1000 == 999:    # every 1000 mini-batches...

            # ...log the running loss
            writer.add_scalar('training loss',
                            running_loss / 1000,
                            epoch * len(data['train'] + i)




    ## Merge
     #   merged = torch.cat([word_embeds.view(), char.view()], dim=2)
