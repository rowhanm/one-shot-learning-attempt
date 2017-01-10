import numpy as np
import os
from scipy.ndimage import imread
from scipy.spatial.distance import cdist

path_to_script_dir = os.path.dirname(os.path.realpath(__file__))
path_to_all_runs = os.path.join(path_to_script_dir, 'all_runs')
labels_file = 'labels_x.txt'

if __name__ == 'main':
    error = np.zeros(10)
    for n in xrange(1, 21):
        ch = str(n)
        if len(cg) == 1:
            ch = '0' + ch
        error[n-1] = classification_run('run'+ch, LoadImgAsPoints, ModHausdorffDistance, 'cost')
        print "run" + str(n) + ", error " + str(error[n-1]) + "%."
    print "average error = " + str(np.mean(error)) + "%"

def LoadImgAsPoints(file):
    im = imread(file, flatten=True)
    im = ~np.array(im,dtype=np.bool)
    d = np.array(im.nonzero()).T
    return d - d.mean(axis=0)

def ModHausdorffDistance(x,y):
    d = cdist(x,y)
    min_dist_x = d.min(axis=1)
    min_dist_y = d.min(axis=0)
    mean_x = np.mean(min_dist_x)
    mean_y = np.mean(min_dist_y)
    return max(mean_x, mean_y)

def classification_run(folder, f_load, f_cost, ftype="cost"):
    assert ftype in {'cost', 'score'}
    with open(os.path.join(path_to_all_runs, folder, labels_file)) as f:
    pairs = [line.split() for line in f.readlines()]

    test_files, train_files = zip(*pairs)
    answers = list(train_files)
    test_files = sorted(test_files)
    train_files = sorted(train_files)
    num_test = len(test_files)
    num_train = len(train_files)

    test_items = [f_load(os.path.join(path_to_all_runs,f)) for f in test_files]
    train_items = [f_load(os.path.join(path_to_all_runs,f)) for f in train_files]

    cost = np.zeros((num_test,num_train))
    for i in xrange(num_test):
        for j in xrange(num_train):
            cost[i,j] = f_cost(test_items[i], train_items[j])
    if ftype == 'cost':
        YHAT = np.argmin(cost,axis=1)
    elif ftype == 'score':
        YHAT = np.argmax(cost,axis=1)

    correct = 0.0
    for i in xrange(num_test):
        if train_files[YHAT[i]] == answers[i]:
            correct += 1.0
    prob_correct = (correct/num_test) * 100
    return 100 - prob_correct
