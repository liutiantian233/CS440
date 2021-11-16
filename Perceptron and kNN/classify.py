# classify.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/27/2018
# Extended by Daniel Gonzales (dsgonza2@illinois.edu) on 3/11/2020

"""
This is the main entry point for MP5. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.

train_set - A Numpy array of 32x32x3 images of shape [7500, 3072].
            This can be thought of as a list of 7500 vectors that are each
            3072 dimensional.  We have 3072 dimensions because there are
            each image is 32x32 and we have 3 color channels.
            So 32*32*3 = 3072. RGB values have been scaled to range 0-1.

train_labels - List of labels corresponding with images in train_set
example: Suppose I had two images [X1,X2] where X1 and X2 are 3072 dimensional vectors
         and X1 is a picture of a dog and X2 is a picture of an airplane.
         Then train_labels := [1,0] because X1 contains a picture of an animal
         and X2 contains no animals in the picture.

dev_set - A Numpy array of 32x32x3 images of shape [2500, 3072].
          It is the same format as train_set

return - a list containing predicted labels for dev_set
"""

import numpy as np


def trainPerceptron(train_set, train_labels, learning_rate, max_iter):
    # TODO: Write your code here
    # return the trained weight and bias parameters
    w = [0] * len(train_set[0])
    b = 0
    for i in range(max_iter):
        for index, label in enumerate(train_labels):
            y_hat = 0
            if np.dot(w, train_set[index]) + b > 0:
                y_hat = 1
            w += learning_rate * (label - y_hat) * train_set[index]
            b += learning_rate * (label - y_hat)
    return w, b


def classifyPerceptron(train_set, train_labels, dev_set, learning_rate, max_iter):
    # TODO: Write your code here
    # Train perceptron model and return predicted labels of development set
    w, b = trainPerceptron(train_set, train_labels, learning_rate, max_iter)
    return_list = []
    for i in dev_set:
        y_hat = 0
        if np.dot(w, i) + b > 0:
            y_hat = 1
        return_list.append(y_hat)
    return return_list


def classifyKNN(train_set, train_labels, dev_set, k):
    # TODO: Write your code here
    return_list = []
    for i in dev_set:
        temp = []
        sort = {}
        for index, j in enumerate(train_set):
            sort[np.linalg.norm(i - j)] = index
            temp.append(np.linalg.norm(i - j))
        temp = sorted(temp)
        total = 0
        for index in range(k):
            total += train_labels[sort[temp[index]]]
        return_list.append(total > k / 2)
    return return_list
