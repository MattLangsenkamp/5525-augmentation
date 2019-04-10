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

    # augment corpi
    corpus_method_1, y_train_method_1 = am.method_1(orig_corpus.copy(), y_train_orig.copy(), word_vectors)
    corpus_method_2, y_train_method_2 = am.method_2(orig_corpus.copy(), y_train_orig.copy(), word_vectors)
    corpus_method_3, y_train_method_3 = am.method_3(orig_corpus.copy(), y_train_orig.copy(), word_vectors)

    # process data so they are in a form that can be fed to classifiers
    X_orig = dp.process_corpus(orig_corpus)
    X_method_1 = dp.process_corpus(corpus_method_1)
    X_method_2 = dp.process_corpus(corpus_method_2)
    X_method_3 = dp.process_corpus(corpus_method_3)
    X_test = dp.process_corpus(test_corpus)

    # train classifiers on original corpus and all augmented corpi
    classifier_orig = cl.train_classifier(X_orig, y_train_orig)
    classifier_method_1 = cl.train_classifier(X_method_1, y_train_method_1)
    classifier_method_2 = cl.train_classifier(X_method_2, y_train_method_2)
    classifier_method_3 = cl.train_classifier(X_method_3, y_train_method_3)

    # test all classifiers with original test data
    acc_orig, conf_mat_orig = t.test_classifier(classifier_orig, X_test, y_test_orig)
    acc_method_1, conf_mat_method_1 = t.test_classifier(classifier_method_1, X_test, y_test_orig)
    acc_method_2, conf_mat_method_2 = t.test_classifier(classifier_method_2, X_test, y_test_orig)
    acc_method_3, conf_mat_method_3 = t.test_classifier(classifier_method_3, X_test, y_test_orig)

    # visualize and export results 
    vis.acc_bar_plot([acc_orig, acc_method_1, acc_method_2, acc_method_3], 
                        ['Original, Method 1', 'Method 2', 'Method 3'])
    vis.visualize_confusion_matrix(acc_method_1, 'Original')
    vis.visualize_confusion_matrix(acc_method_1, 'Method 1')
    vis.visualize_confusion_matrix(acc_method_1, 'Method 2')
    vis.visualize_confusion_matrix(acc_method_1, 'Method 3')
    