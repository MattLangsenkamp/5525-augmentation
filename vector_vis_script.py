import data_loader as dl
import word_vectors as wv  
import visualization as vis 

if __name__ == "__main__":
    # get original data data
    orig_corpus, y_train_orig = dl.load_train_data()
    test_corpus, y_test_orig = dl.load_test_data()

    # develop word vectors
    word_vectors = wv.get_word_vectors(orig_corpus)
    points = vis.visualize_vectors(word_vectors)
    
    inc = 7
    for x in list(range(-52,30,inc)):
        for y in list(range(-54,10,inc)):
            x_bounds = [x, x+inc]
            y_bounds = [y, y+inc]
            try:
                vis.plot_region(x_bounds, y_bounds, points)
            except:
                pass