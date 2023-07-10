import numpy as np
import sklearn.datasets
import matplotlib.pylab as plt
import sklearn.model_selection
import sklearn.svm
import sklearn.neighbors
import sklearn.naive_bayes
import sklearn.neural_network
import cv2

def class_average(X, y, img_size):
    return [X[y == c].mean(0).reshape(img_size) for c in range(10)]

def cross_validate(clf_type, X, y, cv):
    clf = None
    if clf_type == 'kNN':
        clf = sklearn.neighbors.KNeighborsClassifier()
    elif clf_type == 'MLP':
        clf = sklearn.neural_network.MLPClassifier()
    elif clf_type == 'NaiveBayes':
        clf = sklearn.naive_bayes.GaussianNB()
    else:
        clf = sklearn.svm.SVC()
        
    scores = sklearn.model_selection.cross_validate(
        clf, X, y, cv=cv, return_train_score=True)
    return (scores['train_score'], scores['test_score'])

def to_gray(img):
    return 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]

def load_img(img_number):
    img = np.array(cv2.cvtColor(cv2.imread('test_' + str(img_number) + '.png'), cv2.COLOR_BGR2RGB))
    img = to_gray(img)
    img *= 16. / img.max()
    return img.reshape([64])
    
    

