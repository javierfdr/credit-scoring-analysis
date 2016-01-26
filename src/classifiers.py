from __future__ import division

__author__ = 'javierfdr'

import numpy as np
import csv
from sklearn import cross_validation
from sklearn import svm
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.lda import LDA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
import time
from sklearn.lda import LDA
from sklearn.datasets import load_boston
import csv
import random
import matplotlib.pyplot as plt

def rewrite_dataset_fc(filename):
    dataset = []
    letterToNum = {}
    letterCount = 1

    with open('./data_files/'+filename, 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            dataset.append(row)

    clean_dataset = []
    for row in dataset:
        new_row = []
        for value in row:
            try:
                int(value)
                new_row.append(value)
            except ValueError:
                value #skip
        clean_dataset.append(new_row)

    with open('./data_files/new-'+filename, 'wb') as csvWrite:
        newFileWriter = csv.writer(csvWrite, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)

        for row in clean_dataset:
            newFileWriter.writerow(row)

def read_csv(filename,delim=','):
    return np.loadtxt(open("./data_files/"+filename,"rb"),delimiter=delim,skiprows=1)

def load_dataset(filename,delim=','):

    ds = read_csv(filename,delim)
    X = ds[0:len(ds),0:len(ds[0])-1]
    Y = ds[0:len(ds),len(ds[0])-1]

    # transforming NaN to 0
    X = np.nan_to_num(X)
    X = preprocessing.normalize(X, norm='l2')

    return [X,Y]


def load_and_split_dataset(filename,delim=','):
    [X,Y] = load_dataset(filename,delim)

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, Y, test_size=0.3, random_state=0)
    return [X_train, X_test, y_train, y_test]

def train_lda(filename,delim=','):
    start = time.time()
    [X_train, X_test, y_train, y_test] = load_and_split_dataset(filename,delim)
    clf = LDA()
    clf.fit(X_train, y_train)
    end = time.time()
    print('Training Time: '+str((end - start))+'s')

    y_pred = clf.predict(X_test)

    print np.sum(y_pred == y_test)/len(y_pred)
    return y_pred

def getSVMClassifier():
    return svm.SVC(kernel='rbf', C=10, gamma=10, probability=True)

def getSVMLinearClassifier():
    return svm.SVC(kernel='linear', C=10, probability=True)

def getSimpleRDFClassifier():
    rdf = RandomForestClassifier(max_features = 'auto', max_depth=10)
    return rdf

def getCustomRDFClassifier():
    class RandomForestClassifierWithCoef(RandomForestClassifier):
        def fit(self, *args, **kwargs):
            super(RandomForestClassifierWithCoef, self).fit(*args, **kwargs)
            self.coef_ = self.feature_importances_

    rdf = RandomForestClassifierWithCoef(max_features = 'auto', max_depth=6)
    return rdf


def getLDAClassifier(n_components = 2):
    return LDA(n_components=n_components)


def train_classifier(filename,clf,delim=','):

    [X_train, X_test, y_train, y_test] = load_and_split_dataset(filename,delim)

    start = time.time()
    print('Training Classifier')
    clf = clf.fit(X_train, y_train)
    end = time.time()
    print('Training Time: '+str((end - start))+'s')

    print len(X_train)
    print len(X_test)

    print('Testing Results')
    start = time.time()
    score = clf.score(X_test, y_test)
    end = time.time()
    print score
    print('Testing Time: '+str((end - start))+'s')

    print('Testing Againts Training')
    tscore = clf.score(X_train, y_train)
    print tscore

    return [X_train, X_test, y_train, y_test, score,clf]

def getAccuracy(clf,X,y):
    ypred = clf.predict(X)
    return sum(y==ypred)/len(X)


def getClassificationSquare(y_true,y_pred):
    TP = sum((y_true==1.0)==(y_pred==1.0))
    TN = sum((y_true==2.0)==(y_pred==2.0))
    FP = sum((y_true==2.0)==(y_pred==1.0))
    FN = sum((y_true==1.0)==(y_pred==2.0))

    return [TP,TN,FP,FN]

def plot_confusion_matrix(cm, target_names, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

