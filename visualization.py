import matplotlib.pyplot as plt

def visualize_confusion_matrix(conf_mat, classes,
                          title=None,
                          cmap=plt.cm.Blues):
    
    '''labels = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc',
              'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x',
              'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball',
              'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space',
              'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast',
              'talk.politics.misc', 'talk.religion.misc']'''
    
    labels_less = ['atheism', 'graphics', 'ms-windows.misc',
              'pc.hardware', 'mac.hardware', 'windows.x',
              'forsale', 'autos', 'motorcycles', 'baseball',
              'hockey', 'crypt', 'electronics', 'med', 'space',
              'christian', 'politics.guns', 'politics.mideast',
              'politics.misc', 'religion.misc']
    plt.close('all')
    
    fig = plt.figure(figsize=(10.0, 9.0))
    plt.matshow(conf_mat, cmap = cmap, fignum = 1)
    fig.tight_layout()
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.xticks(list(range(20)),labels_less, rotation=90)
    plt.yticks(list(range(20)),labels_less)
    plt.savefig(title+'.png', bbox_inches = 'tight')
    plt.show()


def acc_bar_plot(acc_list, labels):
    # make par plot
    # save as png
    pass

def visualize_vectors(vectors):
    # use demensionality reduction to get vectors into two or three dimensions
    # save as png
    pass
