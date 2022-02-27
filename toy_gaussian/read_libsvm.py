import os, urllib
import itertools

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split

np.random.seed(42)

LIBSVM_URL = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/"
LIBSVM_DOWNLOAD_FN = {"rcv1"       : "rcv1_train.binary.bz2",
                      "mushrooms"  : "mushrooms",
                      "a1a"        : "a1a",
                      "a2a"        : "a2a",
                      "a6a"        : "a6a",
                      "ijcnn"      : "ijcnn1.t.bz2",
                      "w8a"        : "w8a"}

def load_a6a():
    X, y = load_libsvm("a6a", data_dir=os.getcwd())

    # add bias column
    X = np.c_[np.ones(X.shape[0]), X.toarray()]

    # make binary labels 0/1 instead of -1/1 if needed
    # labels = np.unique(y)

    # y[y==labels[0]] = 0
    # y[y==labels[1]] = 1

    # splits used in experiments
    # splits = train_test_split(X, y, test_size=0.2, shuffle=True, 
                # random_state=9513451)
    # X_train, X_test, Y_train, Y_test = splits
    data = np.c_[y, X]
    return data


def load_libsvm(name, data_dir):
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    fn = LIBSVM_DOWNLOAD_FN[name]
    data_path = os.path.join(data_dir, fn)

    if not os.path.exists(data_path):
        url = urllib.parse.urljoin(LIBSVM_URL, fn)
        print("Downloading from %s" % url)
        urllib.request.urlretrieve(url, data_path)
        print("Download complete.")

    X, y = load_svmlight_file(data_path)
    return X, y


a6a = load_a6a()
np.save("a6a.npy", a6a)

print(a6a.shape)