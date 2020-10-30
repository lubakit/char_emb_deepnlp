
"""
DISCLAMER: the code above uses assignment 2 parts made with Nico, examples from existing projects and explanatory posts, mainly not taking te code directly, but builsing the same structures
including tokenization and pre-processing : http://digtime.cn/articles/222/ai-for-trading-character-level-lstm-in-pytorch-98
character embeddings based on word embeddings from pytorch website, https://towardsdatascience.com/implementing-word2vec-in-pytorch-skip-gram-model-e6bae040d2fb

Some parts made by the project partner Karina Hensel were changed and imported as functions or taken and changed for more compatibility for the planned concatenating
"""



import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import codecs
import math
from utils import *

import matplotlib.pyplot as plt
from pathlib import Path


"""
# Data and hyperparameters in case custom-made embeddings would not work
# embeddings_file = '../Data/embeddings/en/glove.6B.100d.bin'

embeddings_file = '../data/embeddings/en/charemb.txt'
data_dir = '../data/conll2003/en/'
model_dir = '../models/'

#from Karinas code

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

"""

"""From here no pretrained character embeddings are used. Path to the file with sentences from the corpus without labels"""
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
#fron pytorch website, not needed
#word_to_ix = {"hello": 0, "world": 1}
#embeds = nn.Embedding(85, 5)  # 85 chars in vocab, 5 dimensional embeddings
#lookup_tensor = torch.tensor([word_to_ix["hello"]], dtype=torch.long)
#hello_embed = embeds(lookup_tensor)
#print(hello_embed)



#hyperparameters synchronised with those in the word model
N_EPOCHS = 100
LEARNING_RATE = 0.1
REPORT_EVERY = 5
EMBEDDING_DIM = 30
HIDDEN_DIM = 20
BATCH_SIZE = 32
N_LAYERS = 2
max_len = 25 # for padding


#Character LSTM
class LSTMModel(nn.Module):
    def __init__(self,
                 embedding_dim,#number of the dimensions for each character
                 character_set_size,
                 n_layers,
                 hidden_dim,
                 n_classes): #our classification
        super(LSTMModel, self).__init__()
        self.embeddings = nn.Embedding(character_set_size, embedding_dim)#character_set_size stands for characters number
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers = 2, batch_first = True)
        self.linear = nn.Linear(hidden_dim, n_classes)
        self.dropout = nn.Dropout(0.5) #if dropout else None
        self.first = True

    def forward(self, inputs):
        embeds = self.embeddings(inputs)    # shape = [bz, max_len, embed_dim]
        lstm_out, (ht, ct) = self.lstm(embeds)
        out = self.linear(ht[-1])   #ht[-1]
        out = F.log_softmax(out, dim=1)
        if self.first:
            self.first = False
            print('embeds.shape = ', embeds.shape)
            print('lstm_out.shape = ',lstm_out.shape)
            print('ht.shape = ', ht.shape)
            print('out.shape = ', out.shape)
        return out

    # --- auxilary functions ---
    def get_minibatch(minibatchwords, chars, classes, max_len):
        # minibatchwords is a list of dicts
        # max_len - word-len will be padded
        mb_x = torch.stack(
            [F.pad(elem['TENSOR'], pad=(0, max_len - len(elem['TENSOR'])), mode='constant', value=0) for elem in
             minibatchwords])
        mb_y = torch.Tensor([label_to_idx(elem['LANGUAGE'], languages) for elem in minibatchwords])
        return mb_x, mb_y

    def label_to_idx(lan, classes):
        languages_ordered = list(classes)
        languages_ordered.sort()
        return torch.LongTensor([languages_ordered.index(lan)])

    def get_word_length(word_ex):
        return len(word_ex['WORD'])

    def evaluate(dataset, model, eval_batch_size, chars, classes):
        correct = 0
        first = False

        for i in range(0, len(dataset), eval_batch_size):
            minibatchwords = dataset[i:i + eval_batch_size]
            mb_x, mb_y = get_minibatch(minibatchwords, chars, classes, max_len)
            model.zero_grad()
            y_pred = model(mb_x)
            correct += (y_pred.argmax(1) == mb_y).sum().item()

            if first:
                first = False
                print(mb_y)
                print(y_pred)

        return correct * 100.0 / len(dataset)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE,  momentum=0.6)
    loss_function = nn.NLLLoss()


    # --- training loop ---
    for epoch in range(N_EPOCHS):
        total_loss = 0

        #shuffling
        shuffle(trainset)


        # Sort od the raining set according to word-length,
        # so that similar-length words end up near each other
        #

        for i in range(0,len(trainset),BATCH_SIZE):   # 1 --> len(trainset)
            minibatchwords = trainset[i:i+BATCH_SIZE]

            #print(minibatchwords)

            mb_x, mb_y = get_minibatch(minibatchwords, chars, classes, max_len)
            mb_y = mb_y.type(torch.LongTensor)

            # WRITE CODE HERE
            model.zero_grad()
            y_pred = model(mb_x)

            #mb_y = mb_y.unsqueeze(dim=0)
            #out = out.unsqueeze(dim=0)

            #print(mb_y)
            #print(out)


            loss = loss_function(y_pred, mb_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print('epoch: %d, loss: %.4f' % ((epoch+1), total_loss))

        if ((epoch+1) % REPORT_EVERY) == 0:
            train_acc = evaluate(trainset,model,BATCH_SIZE,chars,classes)
            dev_acc = evaluate(data['dev'],model,BATCH_SIZE,chars,classes)
            print('epoch: %d, loss: %.4f, train acc: %.2f%%, dev acc: %.2f%%' %
                  (epoch+1, total_loss, train_acc, dev_acc))


    # --- test ---
    test_acc = evaluate(data['test'],model,BATCH_SIZE,character_map,languages)
    print('test acc: %.2f%%' % (test_acc))


    ## Merge
     #   merged = torch.cat([word_embeds.view(), char_lvl[-1].view()], dim=2)