# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
import numpy as np
import math
from tqdm import tqdm
from collections import Counter
import reader

"""
This is the main entry point for MP4. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

"""
  load_data calls the provided utility to load in the dataset.
  You can modify the default values for stemming and lowercase, to improve performance when
       we haven't passed in specific values for these parameters.
"""


def load_data(trainingdir, testdir, stemming=False, lowercase=False, silently=False):
    print(f"Stemming is {stemming}")
    print(f"Lowercase is {lowercase}")
    train_set, train_labels, dev_set, dev_labels = reader.load_dataset(trainingdir, testdir, stemming,
                                                                       lowercase, silently)
    return train_set, train_labels, dev_set, dev_labels


# Keep this in the provided template
def print_paramter_vals(laplace, pos_prior):
    print(f"Unigram Laplace {laplace}")
    print(f"Positive prior {pos_prior}")


"""
You can modify the default values for the Laplace smoothing parameter and the prior for the positive label.
Notice that we may pass in specific values for these parameters during our testing.
"""


def pretreatment_prob(counter, laplace):
    counter_prob = Counter()
    prob = laplace / sum(counter.values())
    for word in counter:
        counter_prob[word] = counter[word] * prob / laplace
    return prob, counter_prob


def naiveBayes(train_set, train_labels, dev_set, laplace=0.001, pos_prior=0.75, silently=False):
    # Keep this in the provided template
    print_paramter_vals(laplace, pos_prior)

    counter_pos = Counter()
    counter_neg = Counter()
    for sentence_index, sentence in enumerate(train_set):
        for word_index, word in enumerate(sentence):
            if train_labels[sentence_index] == 1:
                counter_pos[word] += 1
            else:
                counter_neg[word] += 1
    prob_pos, counter_prob_pos = pretreatment_prob(counter_pos, laplace)
    prob_neg, counter_prob_neg = pretreatment_prob(counter_neg, laplace)

    yhats = []
    for doc in tqdm(dev_set, disable=silently):
        current_pos, current_neg = 0, 0
        for word in doc:
            if word in counter_pos:
                current_pos += np.log(counter_prob_pos[word])
            else:
                current_pos += np.log(prob_pos)
            if word in counter_neg:
                current_neg += np.log(counter_prob_neg[word])
            else:
                current_neg += np.log(prob_neg)
        current_pos += math.log(pos_prior)
        current_neg += math.log(1 - pos_prior)
        if current_pos > current_neg:
            yhats.append(1)
        else:
            yhats.append(0)
    return yhats


# Keep this in the provided template
def print_paramter_vals_bigram(unigram_laplace, bigram_laplace, bigram_lambda, pos_prior):
    print(f"Unigram Laplace {unigram_laplace}")
    print(f"Bigram Laplace {bigram_laplace}")
    print(f"Bigram Lambda {bigram_lambda}")
    print(f"Positive prior {pos_prior}")


# main function for the bigrammixture model
def bigramBayes(train_set, train_labels, dev_set, unigram_laplace=0.001, bigram_laplace=0.005, bigram_lambda=0.5,
                pos_prior=0.5, silently=False):
    # Keep this in the provided template
    print_paramter_vals_bigram(unigram_laplace, bigram_laplace, bigram_lambda, pos_prior)

    counter_pos_unigram = Counter()
    counter_neg_unigram = Counter()
    counter_pos_bigram = Counter()
    counter_neg_bigram = Counter()
    for sentence_index, sentence in enumerate(train_set):
        for word_index, word in enumerate(sentence):
            if train_labels[sentence_index] == 1:
                counter_pos_unigram[word] += 1
                if word_index != len(sentence) - 1:
                    counter_pos_bigram[(word, sentence[word_index + 1])] += 1
            else:
                counter_neg_unigram[word] += 1
                if word_index != len(sentence) - 1:
                    counter_neg_bigram[(word, sentence[word_index + 1])] += 1
    prob_pos_unigram, counter_prob_pos_unigram = pretreatment_prob(counter_pos_unigram, unigram_laplace)
    prob_neg_unigram, counter_prob_neg_unigram = pretreatment_prob(counter_neg_unigram, unigram_laplace)
    prob_pos_bigram, counter_prob_pos_bigram = pretreatment_prob(counter_pos_bigram, bigram_laplace)
    prob_neg_bigram, counter_prob_neg_bigram = pretreatment_prob(counter_neg_bigram, bigram_laplace)

    yhats = []
    for doc in tqdm(dev_set, disable=silently):
        current_pos, current_neg = 0, 0
        bigram_pos, bigram_neg = 0, 0
        for word_index, word in enumerate(doc):
            if word in counter_pos_unigram:
                current_pos += np.log(counter_prob_pos_unigram[word])
            else:
                current_pos += np.log(prob_pos_unigram)
            if word in counter_neg_unigram:
                current_neg += np.log(counter_prob_neg_unigram[word])
            else:
                current_neg += np.log(prob_neg_unigram)
            if word_index != len(doc) - 1:
                if (word, doc[word_index + 1]) in counter_pos_bigram:
                    bigram_pos += np.log(counter_prob_pos_bigram[(word, doc[word_index + 1])])
                else:
                    bigram_pos += np.log(prob_pos_bigram)
                if (word, doc[word_index + 1]) in counter_neg_bigram:
                    bigram_neg += np.log(counter_prob_neg_bigram[(word, doc[word_index + 1])])
                else:
                    bigram_neg += np.log(prob_neg_bigram)
        bigram_pos += math.log(pos_prior)
        bigram_neg += math.log(1 - pos_prior)
        current_pos += math.log(pos_prior)
        current_neg += math.log(1 - pos_prior)
        total_pos = bigram_lambda * bigram_pos + (1 - bigram_lambda) * current_pos
        total_neg = bigram_lambda * bigram_neg + (1 - bigram_lambda) * current_neg
        if total_pos > total_neg:
            yhats.append(1)
        else:
            yhats.append(0)
    return yhats
