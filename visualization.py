import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator

def visualize_confusion_matrix(conf_mat, classes,
                          title=None,
                          cmap=plt.cm.Blues):

    
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

    plt.set_cmap(plt.cm.Blues)    

    fig = plt.figure(figsize=(10.0, 9.0))
    ax = fig.add_subplot(111)

    ax.xaxis.set_major_formatter(FuncFormatter(format_fn))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.bar(range(len(acc_list)), acc_list)
    plt.xlabel('Method')
    plt.ylabel('Accuracy')
    plt.savefig('bar.png', bbox_inches = 'tight')
    plt.show()

def visualize_vectors(vectors):
    # use demensionality reduction to get vectors into two or three dimensions
    # save as png
    pass

def format_fn(tick_val, tick_pos):
    labels = ['Original', 'Method 1', 'Method 2', 'Method 3']
    print(tick_val)
    if int(tick_val) in [0,1,2,3]:
        return labels[int(tick_val)]
    else:
        return ''
