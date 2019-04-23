import augmentation_methods as am
import data_loader as dl
import word_vectors as wv  
import data_preprocessing as dp 
import classifier as cl   
import testing as t
import visualization as vis 

if __name__ == "__main__":
    # get original data data
    orig_corpus, y_train_orig = dl.load_train_data()
    test_corpus, y_test_orig = dl.load_test_data()

    # develop word vectors
    word_vectors = wv.get_word_vectors(orig_corpus)
    points = vis.visualize_vectors(word_vectors)