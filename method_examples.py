import augmentation_methods as am
import data_loader as dl
import word_vectors as wv  


if __name__ == "__main__":
    # get original data in tokenized form
    orig_corpus, y_train_orig = dl.load_train_data()
    test_corpus, y_test_orig = dl.load_test_data()

    # develop word vectors
    word_vectors = wv.get_word_vectors(orig_corpus)
    
        # augment corpi
    corpus_method_1, y_train_method_1 = am.method_1([orig_corpus[89]], [y_train_orig[89]], word_vectors)
    corpus_method_2, y_train_method_2 = am.method_2([orig_corpus[89]], [y_train_orig[89]], word_vectors)
    corpus_method_3, y_train_method_3 = am.method_3([orig_corpus[89]], [y_train_orig[89]], word_vectors)
