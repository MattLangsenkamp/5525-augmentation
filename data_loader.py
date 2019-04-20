from sklearn.datasets import fetch_20newsgroups
from nltk import word_tokenize
import numpy as np
from nltk.corpus import stopwords
import re

# returns list of all data in memory or a generator if dataset is too large 
def load_train_data():
    news = fetch_20newsgroups(subset='train', shuffle=True, random_state=42, remove=('headers', 'footers'), download_if_missing=True)
    orig_corpus_sents = news.data
    orig_corpus = []
    stop_words = set(stopwords.words('english'))
    
    for i in range(len(orig_corpus_sents)):
        new_sent = re.sub("['\"-]","", orig_corpus_sents[i])
        new_sent = re.sub("[^a-zA-Z]"," ", new_sent)
        word_tokens = word_tokenize(new_sent)
        filtered_sentence = [w for w in word_tokens if not w in stop_words] 
        
        filtered_sentence_no_punc = []
        for w in filtered_sentence:
            if len(w) > 1:
                filtered_sentence_no_punc.append(w.lower())
        orig_corpus.append(filtered_sentence_no_punc)

        
    return np.array(orig_corpus), news.target

def load_test_data():
    # TODO should return a list of tokenized sentences
    # TODO remove stop words 
    return None, None