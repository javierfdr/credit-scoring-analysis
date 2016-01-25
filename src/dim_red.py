__author__ = 'javierfdr'

import six
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition
import numpy as np
from sklearn.decomposition import PCA
from sklearn.lda import LDA
from sklearn import datasets
from matplotlib import colors
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from random import randint
from sklearn.linear_model import RandomizedLasso
from sklearn.cross_validation import cross_val_score, ShuffleSplit
import random

import time

def plot1D(X,y,names = []):

    plt.figure()
    classes = np.unique(y)
    colors_ = list(six.iteritems(colors.cnames))
    hex_ = [color[1] for color in colors_]
    rgb = [colors.hex2color(color) for color in hex_]
    colors_ =[]

    for i in range(0, len(classes)):
        colors_.append(rgb[i])

    if (len(names) == 0):
        names = classes

    for c, i, target_name in zip(colors_, classes , names):
        plt.scatter(X[y == i, 0],np.zeros(len(X[y == i, 0])), c=c, label=target_name)
    plt.legend()
    plt.show()

def plot2D(X,y,names = []):

    plt.figure()
    classes = np.unique(y)
    colors_ = list(six.iteritems(colors.cnames))
    hex_ = [color[1] for color in colors_]
    rgb = [colors.hex2color(color) for color in hex_]
    colors_ =[]

    for i in range(0, len(classes)):
        colors_.append(rgb[i])

    if (len(names) == 0):
        names = classes

    for c, i, target_name in zip(colors_, classes , names):
        plt.scatter(X[y == i, 0], X[y == i, 1], c=c, label=target_name)
    plt.legend()
    plt.show()

def plotPCA3D(X,y,names=[]):

    plt.cla()
    pca = decomposition.PCA(n_components=3)
    pca.fit(X)
    X = pca.transform(X)

    fig = plt.figure(1, figsize=(4, 3))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

    classes = np.unique(y)
    colors_ = list(six.iteritems(colors.cnames))

    hex_ = [color[1] for color in colors_]
    rgb = [colors.hex2color(color) for color in hex_]
    class_label = []
    for i in range(0,len(classes)):
        colors_.append(rgb[i])

        if(len(names) == 0):
            class_label.append((str(i),i))
        else:
            class_label.append((names[i],i))


    for name, label in class_label:
        ax.text3D(X[y == label, 0.0].mean(),
                  X[y == label, 1.0].mean() + 1.5,
                  X[y == label, 2.0].mean(), name,
                  horizontalalignment='center',
                  bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
    # Reorder the labels to have colors matching the cluster results
    y = y.astype(int)
    #y = np.choose(y, class_label)
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.hot)

    x_surf = [X[:, 0].min(), X[:, 0].max(),
              X[:, 0].min(), X[:, 0].max()]
    y_surf = [X[:, 0].max(), X[:, 0].max(),
              X[:, 0].min(), X[:, 0].min()]
    x_surf = np.array(x_surf)
    y_surf = np.array(y_surf)
    v0 = pca.transform(pca.components_[[0]])
    v0 /= v0[-1]
    v1 = pca.transform(pca.components_[[1]])
    v1 /= v1[-1]

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])

    plt.show()

def plotLDA3D(X,y,names = []):

    plt.cla()
    lda = LDA(n_components=3)
    lda.fit(X,y)
    X = lda.transform(X)

    fig = plt.figure(1, figsize=(4, 3))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

    classes = np.unique(y)
    colors_ = list(six.iteritems(colors.cnames))
    hex_ = [color[1] for color in colors_]
    rgb = [colors.hex2color(color) for color in hex_]
    colors_ =[]

    class_label = []
    for i in range(0, len(classes)):
        colors_.append(rgb[i])

        if(len(names) == 0):
            class_label.append((str(i),i))
        else:
            class_label.append((names[i],i))

    for name, label in class_label:
        ax.text3D(X[y == label, 0.0].mean(),
                  X[y == label, 1.0].mean() + 1.5,
                  X[y == label, 2.0].mean(), name,
                  horizontalalignment='center',
                  bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
    # Reorder the labels to have colors matching the cluster results
    y = y.astype(int)
    #y = np.choose(y, class_label)
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.hot)

    x_surf = [X[:, 0].min(), X[:, 0].max(),
              X[:, 0].min(), X[:, 0].max()]
    y_surf = [X[:, 0].max(), X[:, 0].max(),
              X[:, 0].min(), X[:, 0].min()]
    x_surf = np.array(x_surf)
    y_surf = np.array(y_surf)
    v0 = lda.transform(lda.coef_[[0]])
    v0 /= v0[-1]
    v1 = lda.transform(lda.coef_[[1]])
    v1 /= v1[-1]

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])

    plt.show()

def plotPCA(X,y,names = []):

    pca = PCA(n_components=2)
    X_r = pca.fit(X).transform(X)

    plot2D(X_r,y,names)


def plotLDA(X,y,names=[]):

    lda = LDA(n_components=2)
    X_r = lda.fit(X,y).transform(X)

    plot2D(X_r,y,names)

def plotLDA1D(X,y,names=[]):

    lda = LDA(n_components=2)
    X_r = lda.fit(X,y).transform(X)

    plot1D(X_r,y,names)

def plotSVM(clf, X,y):
    h = 0.2
    x_min, x_max = X[:,0].min() -1 , X[:,0].max() + 1
    y_min, y_max = X[:,1].min() -1 , X[:,1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx,yy,Z,cmap=plt.cm.Paired, alpha = 0.8)
    plt.scatter(X[:,0],X[:,1],c=y,cmap=plt.cm.Paired)
    plt.xlim(xx.min(), yy.max())
    plt.ylim(xx.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.show()

def recursive_fs(X,y,clf,num_features):
    # create the RFE model and select 3 attributes
    rfe = RFE(clf, num_features)

    start = time.time()
    rfe = rfe.fit(X, y)
    # summarize the selection of the attributes
    end = time.time()
    print('Training Time: '+str((end - start))+'s')
    return rfe

def recursive_fs_cv(X,y,clf):
    # create the RFE model and select 3 attributes
    rfe = RFECV(clf, step=1, cv=5)

    start = time.time()
    rfe = rfe.fit(X, y)
    # summarize the selection of the attributes
    end = time.time()
    print('Training Time: '+str((end - start))+'s')
    return rfe


def transform_rfe_and_plot(rfe,X,y):
    X_r = rfe.transform(X)
    plotPCA3D(X_r,y)

def iter_relief_weights(X,y,prop):
    w = np.zeros(X.shape[1])
    ii32 = np.iinfo(np.int32)

    num_features = X.shape[1]

    for i in range(0,int((X.shape[0]*prop))):

        rand_index = randint(0,X.shape[0]-1)
        example = X[rand_index]
        min_dist_same = ii32.max
        min_dist_different = ii32.max
        closest_same = 0
        closest_different = 0

        # same class
        for same_class_e in X[y==(y[rand_index])]:
            dist = np.linalg.norm(same_class_e-example)

            # same example
            if(dist == 0.0):
                continue

            if(dist < min_dist_same):
                min_dist_same = dist
                closest_same = same_class_e

        # different class
        for different_class_e in X[y!=(y[rand_index])]:
            dist = np.linalg.norm(different_class_e-example)

            # same example
            if(dist == 0.0):
                continue

            if(dist < min_dist_different):
                min_dist_different = dist
                closest_different = same_class_e

        # updating relief weights
        for k in range(0,num_features):
            same_feat_dist = np.linalg.norm(closest_same[k]-example[k])
            different_feat_dist = np.linalg.norm(closest_different[k]-example[k])

            w[k] = w[k] - (same_feat_dist/num_features) + (different_feat_dist/num_features)

    return w

def relief_wrapper(X_train,y_train, X_test, y_test,prop,clf):

    start = time.time()
    deleted_features = []
    total_features = X_train.shape[1]
    clf.fit(X_train,y_train)
    best_score = clf.score(X_test,y_test)

    print 'Initial Score: '+str(best_score)

    for i in range(0,total_features):
        print i
        w = iter_relief_weights(X_train,y_train,prop)
        min_index = np.argmin(w)
        X_train2 = np.delete(X_train,min_index,1) #delete min_index column
        X_test2 = np.delete(X_test,min_index,1)

        clf.fit(X_train2,y_train)
        new_score = clf.score(X_test2,y_test)

        if(new_score > best_score):
            best_score = new_score
            X_train = np.delete(X_train,min_index,1)
            X_test = np.delete(X_test,min_index,1)
            deleted_features.append(min_index)
        else:
            print 'New score is not enough: '+str(new_score)
            break

    end = time.time()
    print 'Final Score: '+str(best_score)

    print('Training Time: '+str((end - start))+'s')
    return [deleted_features]

def lasso_fs(X,y):
    rlasso = RandomizedLasso()
    rlasso.fit(X, y)
    classes = range(0,X.shape[1])

    print "Features sorted by their score:"
    print sorted(zip(map(lambda x: round(x, 4), rlasso.scores_),classes), reverse=True)

def feature_analysis_f1(X,y,clf):
    scores = []
    names = range(0,X.shape[1])
    for i in range(X.shape[1]):
         score = cross_val_score(clf, X[:, i:i+1], y, scoring="f1_weighted",
                                  cv=ShuffleSplit(len(X), 3, .3))
         scores.append((round(np.mean(score), 3), names[i]))
    print sorted(scores, reverse=True)

def feature_analysis_prec(X,y,clf):
    scores = []
    names = range(0,X.shape[1])
    for i in range(X.shape[1]):
         score = cross_val_score(clf, X[:, i:i+1], y, scoring="precision",
                                  cv=ShuffleSplit(len(X), 3, .3))
         scores.append((round(np.mean(score), 3), names[i]))
    print sorted(scores, reverse=True)

def graph_plot(accuracy_list):
    s = accuracy_list
    t = range(1,len(accuracy_list)+1)
    plt.plot(t, s)

    plt.xlabel('Number of features')
    plt.ylabel('Accuracy')
    plt.title('RDF accuracy on different number of selected best features')
    plt.grid(True)
    plt.show()

