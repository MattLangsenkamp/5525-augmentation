import numpy as np
import random as r
import nltk
from itertools import chain

# returns augmented corpus of documents
def method_1(corpus, y, word_vectors):
    augmented_data = []
    augmented_y = []
    for i in range(len(corpus)):
        sample = corpus[i]
        new_sample = sample.copy()
        indices = pick_word_indices(sample, word_vectors)
        for index in indices:
            new_sample[index] = word_vectors.most_similar(sample[index])[0][0]
            
        augmented_data.append(sample)
        augmented_y.append(y[i])

        if len(indices) > 0:
            augmented_data.append(new_sample)
            augmented_y.append(y[i])

    return augmented_data, augmented_y

# returns augmented corpus of documents
def method_2(corpus, y, word_vectors):
    augmented_data = []
    augmented_y = []
    for i in range(len(corpus)):
        sample = corpus[i].copy()
        indices = pick_word_indices(sample, word_vectors)
        for index in indices:
            new_sample = list(chain(sample[:index+1], [word_vectors.most_similar(sample[index])[0][0]], sample[index + 1:]))
        augmented_data.append(new_sample)
        augmented_y.append(y[i])

    return augmented_data, augmented_y


# returns augmented corpus of documents
def method_3(corpus, y, word_vectors):
    augmented_data = []
    augmented_y = []
    for i in range(len(corpus)):
        sample = corpus[i]
        new_sample = sample.copy()
        indices = pick_word_indices_verb_or_adjective(sample, word_vectors)
        for index in indices:
            new_sample[index] = word_vectors.most_similar(sample[index])[0][0]
        
        augmented_data.append(sample)
        augmented_y.append(y[i])

        if len(indices) > 0:
            augmented_data.append(new_sample)
            augmented_y.append(y[i])

    return augmented_data, augmented_y


def pick_word_indices(sample, word_vectors):
    
    valid_indices = []
    for i in range(len(sample)):
        if sample[i] in word_vectors.wv.vocab:
            valid_indices.append(i)
    
    if len(valid_indices)>0:
        amount_words_swapped = r.randint(1, max(len(valid_indices)-1, 1))
        return r.sample(valid_indices, amount_words_swapped)
    else:
        return []

def pick_word_indices_verb_or_adjective(sample, word_vectors):
    
    tags = nltk.pos_tag(sample)
    valid_indices = []
    for i in range(len(tags)):
        if ('VB' in tags[i][1] or 'NN' in tags[i][1]) and tags[i][0] in word_vectors.wv.vocab:
            valid_indices.append(i)

    if len(valid_indices)>0:
        amount_words_swapped = r.randint(1, max(len(valid_indices)-1, 1))
        return r.sample(valid_indices, amount_words_swapped)
    else:
        return []