from sklearn.feature_extraction.text import TfidfVectorizer


def process_corpus_orig(corpus):
    # TODO convert to tfidf
    vectorizer = TfidfVectorizer(sublinear_tf = 'true')
    X = vectorizer.fit_transform([' '.join(x) for x in corpus])

    return X, vectorizer

def process_corpus(corpus, vectorizer):
    # TODO convert to tfidf
    X = vectorizer.transform([' '.join(x) for x in corpus])

    return X