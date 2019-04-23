import gensim.models.word2vec as w2v
import multiprocessing

# return word gensim word2vec object. it has methods for obtaining 
# the most similat word and the entire coprpus etc.
def get_word_vectors(corpus):

    num_features = 300
    min_word_count = 70
    num_workers = multiprocessing.cpu_count()
    context_size = 7
    downsampling = 1e-5
    seed = 1

    vecs = w2v.Word2Vec(
        sg=1,
        seed=seed,
        workers=num_workers,
        size=num_features,
        min_count=min_word_count,
        window=context_size,
        sample=downsampling
    )

    vecs.build_vocab(corpus)
    vecs.train(corpus, total_examples = len(corpus), epochs = 100)

    return vecs


