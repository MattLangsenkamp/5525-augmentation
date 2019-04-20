from sklearn.feature_extraction.text import TfidfVectorizer


def process_corpus(corpus):
    # TODO convert to tfidf
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform([''.join(x) for x in corpus])

    return X