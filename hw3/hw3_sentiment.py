# STEP 1: rename this file to hw3_sentiment.py

# feel free to include more imports as needed here
# these are the ones that we used for the base model
import random
import numpy as np
import sys
from collections import Counter
import math
import nltk

"""
Your name and file comment here: Kelley Kelley
"""

"""
Cite your sources here: Lecture Slides
"""

"""
Implement your functions that are not methods of the Sentiment Analysis class here
"""


def generate_tuples_from_file(training_file_path):
    file = open(training_file_path, 'r')
    text = file.read()
    file.close()
    lines = text.split('\n')
    tuples = []
    for li in lines:
        if not li == '':
            split = li.split('\t')
            if not len(split) == 3:
                print(split)
            tuples.append((split[0], split[1], split[2]))
    return tuples


def precision(gold_labels, classified_labels):
    true_pos = 0
    false_pos = 0
    for i in range(len(gold_labels)):
        if (gold_labels[i] == '1') and (classified_labels[i] == '1'):
            true_pos += 1
        if (gold_labels[i] == '0') and (classified_labels[i] == '1'):
            false_pos += 1
    prec = true_pos / (true_pos + false_pos)
    return prec


def recall(gold_labels, classified_labels):
    true_pos = 0
    false_neg = 0
    for i in range(len(gold_labels)):
        if (gold_labels[i] == '1') and (classified_labels[i] == '1'):
            true_pos += 1
        if (gold_labels[i] == '1') and (classified_labels[i] == '0'):
            false_neg += 1
    rec = true_pos / (true_pos + false_neg)
    return rec


def f1(gold_labels, classified_labels):
    f_1 = 2 * precision(gold_labels, classified_labels) * recall(gold_labels, classified_labels)
    f_1 /= (precision(gold_labels, classified_labels) + recall(gold_labels, classified_labels))
    return f_1


"""
Implement any other non-required functions here
"""

"""
implement your SentimentAnalysis class here
"""


class SentimentAnalysis:

    def __init__(self):
        # do whatever you need to do to set up your class here
        self.positive = {}
        self.negative = {}
        self.negative_words = 0
        self.positive_words = 0
        self.positive_count = 0
        self.negative_count = 0
        self.total_count = 0
        self.total = {}

    def train(self, examples):
        for ex in examples:
            if not ex == '':
                self.total_count += 1
                if ex[2] == '1':
                    self.positive_count += 1
                else:
                    self.negative_count += 1
                words = ex[1].split(' ')
                for word in words:
                    if not word == '':
                        if word in self.total:
                            self.total[word] += 1
                        else:
                            self.total[word] = 1
                        if ex[2] == '0':
                            self.negative_words += 1
                            if word in self.negative:
                                self.negative[word] += 1
                            else:
                                self.negative[word] = 1
                        elif ex[2] == '1':
                            self.positive_words += 1
                            if word in self.positive:
                                self.positive[word] += 1
                            else:
                                self.positive[word] = 1

    def score(self, data):
        negative = 0
        positive = 0
        words = data.split(' ')
        for w in words:
            if w in self.total:
                if w in self.positive:
                    prob = self.positive[w] + 1
                    prob /= (self.positive_words + len(self.total))
                    prob = np.log(prob)
                    positive += prob
                else:
                    prob = 1/(self.positive_words + len(self.total))
                    prob = np.log(prob)
                    positive += prob
                if w in self.negative:
                    prob = self.negative[w] + 1
                    prob /= (self.negative_words + len(self.total))
                    prob = np.log(prob)
                    negative += prob
                else:
                    prob = 1/(self.negative_words + len(self.total))
                    prob = np.log(prob)
                    negative += prob
        negative += np.log(self.negative_count/self.total_count)
        positive += np.log(self.positive_count/self.total_count)
        negative = np.exp(negative)
        positive = np.exp(positive)
        return {'0': negative, '1': positive}


    def classify(self, data):
        dictionary = self.score(data)
        if dictionary['0'] >= dictionary['1']:
            return '0'
        else:
            return '1'

    def featurize(self, data):
        words = data.split(' ')
        tup = []
        for w in words:
            tup.append((w, 'True'))
        return tup

    def __str__(self):
        return "Naive Bayes - bag-of-words baseline"


class SentimentAnalysisImproved:

    def __init__(self):
        # do whatever you need to do to set up your class here
        self.weights = {}
        self.counts = {}
        self.bias = 0.0
        self.step_size = .00001
        self.negative = {}
        self.positive = {}

    def train(self, examples):
        for ex in examples:
            words = ex[1].split(' ')
            for word in words:
                if not word in self.weights:
                    self.weights[word] = random.uniform(-.0001, .0001)
        for word in self.weights:
            self.counts[word] = 0
            self.negative[word] = 0
            self.positive[word] = 0
        for ex in examples:
            words = ex[1].split(' ')
            for word in words:
                if ex[2] == '0':
                    self.positive[word] += 1
                else:
                    self.negative[word] += 1
        for i in range(500):

            sig = 0
            words = ex[1].split(' ')
            for word in words:
                sig += self.weights[word] * self.positive[word]
            c_pos = 1/(1 + np.exp(-(sig + self.bias)))
            gradients = {}
            for word in words:
                gradients[word] = self.positive[word] * (c_pos - 1)
            for word in words:
                self.weights[word] = self.weights[word] - (self.step_size * gradients[word])

            sig = 0
            for word in words:
                sig += self.weights[word] * self.negative[word]
            c_neg = 1/(1 + np.exp(-(sig)))
            gradients = {}
            for word in words:
                gradients[word] = self.negative[word] * (c_neg)
            for word in words:
                self.weights[word] = self.weights[word] - (self.step_size * gradients[word])

            '''
            sig = 0
            for word in words:
                sig += self.weights[word] * self.counts[word]
            c = 1/(1 + np.exp(-(sig + self.bias)))
            gradients = {}
            if ex[2] == '0':
                for word in words:
                    gradients[word] = self.counts[word] * c
            elif ex[2] == '1':
                for word in words:
                    gradients[word] = self.counts[word] * (c - 1)
            for word in words:
                self.weights[word] = self.weights[word] - (self.step_size * gradients[word])
            '''


    def score(self, data):
        words = data.split(' ')
        self.counts.clear()
        for word in words:
            if word in self.weights:
                if word in self.counts:
                    self.counts[word] += 1
                else:
                    self.counts[word] = 1
        sig = 0
        for word in words:
            if word in self.weights:
                sig += self.weights[word] * self.counts[word]
        sig += self.bias
        p1 = np.exp(-sig)
        p1 = 1/(1 + p1)
        p0 = 1 - p1
        '''
        print(p0, p1)
        if p0 >= p1:
            print('p0')
        '''
        return {'0': p1, '1': p0}

    def classify(self, data):
        dictionary = self.score(data)
        if dictionary['0'] >= dictionary['1']:
            return '0'
        else:
            return '1'

    def featurize(self, data):
        words = data.split(' ')
        tup = []
        for w in words:
            tup.append((w, 'True'))
        return tup

    def __str__(self):
        return "Improved Classifier"


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage:", "python hw3_sentiment.py training-file.txt testing-file.txt")
        sys.exit(1)

    training = sys.argv[1]
    testing = sys.argv[2]

    sa = SentimentAnalysis()
    print(sa)
    # do the things that you need to with your base class
    tups = generate_tuples_from_file(training)
    sa.train(tups)

    tups = generate_tuples_from_file(testing)
    classified = []
    labels = []
    for line in tups:
        classified.append(sa.classify(line[1]))
        labels.append(line[2])
    print('Precision =', precision(labels, classified))
    print('Recall =', recall(labels, classified))
    print('f1 =', f1(labels, classified))

    improved = SentimentAnalysisImproved()
    print(improved)
    # do the things that you need to with your improved class
    tups = generate_tuples_from_file(training)
    improved.train(tups)

    tups = generate_tuples_from_file(testing)
    classified = []
    labels = []
    for line in tups:
        classified.append(improved.classify(line[1]))
        labels.append(line[2])
    print('Precision =', precision(labels, classified))
    print('Recall =', recall(labels, classified))
    print('f1 =', f1(labels, classified))