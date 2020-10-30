import torch
import torch as nn
import codecs

""" Utility functions to load the data sets and preprocess the input """

"""From the Karinas code in the attempt to merge to models, changed so that it suits character embeddings also. """

TAG_INDICES = {'I-PER':0, 'B-PER':1, 'I-LOC':2, 'B-LOC':3, 'I-ORG':4, 'B-ORG':5, 'I-MISC':6, 'B-MISC':7, 'O':8}

def load_sentences(corpus_file):


    sentences, tok, pos, chunk, ne = [], [], [], [], []

    with open(corpus_file, 'r') as f:
        for line in f.readlines():
            if line == ('-DOCSTART- -X- -X- O\n') or line == '\n':
                # pass
                # if line=='\n':
                # Sentence as a sequence of tokens, POS, chunk and NE tags
                sentence = dict({'TOKENS': [], 'POS': [], 'CHUNK_TAG': [], 'NE': [], 'SEQ': [], 'CHAR': []})
                sentence['TOKENS'] = tok
                sentence['POS'] = pos
                sentence['CHUNK_TAG'] = chunk
                sentence['NE'] = ne
                sentence['CHAR']:[c for c in tok] = ch

                # Once a sentence is processed append it to the list of sentences
                sentences.append(sentence)

                # Reset sentence information
                tok = []
                pos = []
                chunk = []
                ne = []
                ch = []
            else:
                l = line.split(' ')

                # Append info for next word
                tok.append(l[0])
                pos.append(l[1])
                chunk.append(l[2])
                ne.append(l[3].strip('\n'))
                ch.append(l[4])

    return sentences

def word_char_dicts(wor):
    """
    Create a dictionary of all words in the file:
        
        sentences - list of sentence dictionaries.

    Returns - set of unique tokens

    """
    
    words = set()
    
    for s in wor:
        words.update(s['TOKENS'])
    
    # Create character dictionary
    separator = ', '
    chars = set(separator.join(words))
    words.update('PAD')
    words.update('UNK')
    return words, chars
    

def ne_labels_dict(sentences):
    """
    Create a dictionary of all NE labels in the file

    sentences - list of sentence dictionaries.

    Returns - Set of unique NE labels

    """
    
    labels = set()
    
    for s in sentences:
        labels.update(s['NE'])
    
    return labels

#def build_vocab(sents, labels):
    
def read_conll_datasets(data_dir):
    data = {}
    for data_set in ["train","test","valid"]:
        data[data_set] = load_sentences("%s/%s.txt" % (data_dir,data_set)) 
        
    #words = word_char_dicts(data["train"])[0]
    return data#, words




# =========================
# Test
#sents = load_sentences('../Data/conll2003/en/test.txt')
#print(sents)
#tokens = word_char_dicts(sents)
#labels = ne_labels_dict(sents)
#print(labels)  
#data = read_conll_datasets('../Data/conll2003/en')
#print(data['train'])