import sys
import pandas as pd
import scipy
import sklearn
import nltk
import numpy as np
from gensim.models import Word2Vec
import random

from collections import Counter

"""
Your name and file comment here:
Kelley Kelley
"""

"""
Cite your sources here:
"""


def generate_tuples_from_file(file_path):
    """
  Implemented for you. 

  counts on file being formatted like:
  1 Comparison  O
  2 with  O
  3 alkaline  B
  4 phosphatases  I
  5 and O
  6 5 B
  7 - I
  8 nucleotidase  I
  9 . O

  1 Pharmacologic O
  2 aspects O
  3 of  O
  4 neonatal  O
  5 hyperbilirubinemia  O
  6 . O

  params:
    file_path - string location of the data
  return:
    a list of tuples in the format [(token, label), (token, label)...]
  """
    current = []
    f = open(file_path, "r", encoding="utf8")
    examples = []
    for line in f:
        if len(line.strip()) == 0 and len(current) > 0:
            examples.append(current)
            current = []
        else:
            pieces = line.strip().split()
            current.append(tuple(pieces[1:]))
    if len(current) > 0:
        examples.append(current)
    f.close()
    return examples


def get_words_from_tuples(examples):
    """
  You may find this useful for testing on your development data.

  params:
    examples - a list of tuples in the format [[(token, label), (token, label)...], ....]
  return:
    a list of lists of tokens
  """
    return [[t[0] for t in example] for example in examples]


def decode(data, probability_table, pointer_table):
    """
  TODO: implement
  params: 
    data - a list of tokens
    probability_table - a list of dictionaries of states to probabilities, 
      one dictionary per word in the test data that represents the
      probability of being at that state for that word
    pointer_table - a list of dictionaries of states to states, 
      one dictionary per word in the test data that represents the 
      backpointers for which previous state led to the best probability
      for the current state
  return:
    a list of tuples in the format [(token, label), (token, label)...]
  """
    # this is the list of tuples we will return
    tuples = []

    i = len(probability_table) - 1
    max = -1
    prev = None
    for state in probability_table[i]:
        if probability_table[i][state] > max:
            max = probability_table[i][state]
            prev = state
    ptr = pointer_table[i][prev]
    tuples.insert(0, (data[i], prev))
    i -= 1
    while not ptr == None:
        tuples.insert(0, (data[i], ptr))
        ptr = pointer_table[i][ptr]
        i -= 1
    return tuples


def precision(gold_labels, classified_labels):
    """
  TODO: implement
  true pos/(true pos + false pos)
  params:
    gold_labels - a list of tuples in the format [(token, label), (token, label)...]
    classified_labels - a list of tuples in the format [(token, label), (token, label)...]
  return:
    float value of precision at the entity level
  """
    if not len(gold_labels) == len(classified_labels):
        print(len(gold_labels), len(classified_labels))
    # this variable stores how many true positives we have
    true_pos = 0
    # this variable stores how many false positives we have
    false_pos = 0
    end = len(gold_labels) - 1
    for i in range(len(gold_labels)):
        # if our token is labeled B by both, then we have an entity that could
        # potentially be a true pos
        if gold_labels[i][1] == 'B' and classified_labels[i][1] == 'B':
            # store our i because we are about to increment it
            # to get our entire entity
            i_pos = i
            i_beg = i
            # this is how we are going to store our gold label entity
            gold_entity = [gold_labels[i]]
            # this will grab our entire entity by iterating until
            # we don't have an I
            while not (end <= i):
                if not gold_labels[i + 1][1] == 'I':
                  break;
                i += 1
                # add each tupel to our gold_entity
                gold_entity.append(gold_labels[i])
            # now we are going to get the entire labeled entity
            # the same way we got our gold entity
            labeled_entity = [classified_labels[i_pos]]
            if classified_labels[i_pos][1] == 'B':
                while not (end <= i_pos):
                    if not classified_labels[i_pos + 1][1] == 'I':
                      break;
                    i_pos += 1
                    labeled_entity.append(classified_labels[i_pos])
            # if our two entities are the same, then it means the
            # entity was correctly labeled from end to end
            if gold_entity == labeled_entity:
                true_pos += 1
            # if not then it is considered a false positive since
            # the labels had a B and/or multiple I's that didn't
            # end up equal to our gold entity
            else:
                false_pos += 1
            # reset our i so we don't accidently skip over anything
            # probably should have just iterated i_beg or something instead
            # but it works so doesn't particularly matter now
            i = i_beg
        # if we labeled something B and it wasn't supposed to be then
        # it is a false positive
        elif classified_labels[i][1] == 'B':
            false_pos += 1
    # testing
    # print(true_pos)
    # print(false_pos)
    # do the math and return the value
    if true_pos + false_pos == 0:
        return 0.0
    return true_pos / (true_pos + false_pos)


def recall(gold_labels, classified_labels):
    """
    TODO: implement
    true pos/(true pos + false neg)
    params:
      gold_labels - a list of tuples in the format [(token, label), (token, label)...]
      classified_labels - a list of tuples in the format [(token, label), (token, label)...]
    return:
      float value of recall at the entity level
    """

    # this variable stores how many true positives we have
    true_pos = 0
    # this variable stores how many false positives we have
    false_neg = 0
    for i in range(len(gold_labels)):
        # if gold labels labeled it B then we could have a true_pos or
        # a false_neg
        if gold_labels[i][1] == 'B':
            # store our i because we are about to increment it
            # to get our entire entity
            i_pos = i
            i_beg = i
            # this is how we are going to store our gold label entity
            gold_entity = [gold_labels[i]]
            # this will grab our entire entity by iterating until
            # we don't have an I
            end = len(gold_labels) - 1
            while not (end <= i):
                if not gold_labels[i + 1][1] == 'I':
                  break;
                i += 1
                # add each tupel to our gold_entity
                gold_entity.append(gold_labels[i])
            # now we are going to get the entire labeled entity
            # the same way we got our gold entity
            labeled_entity = [classified_labels[i_pos]]
            if classified_labels[i_pos][1] == 'B':
                while not (end <= i_pos):
                    if not classified_labels[i_pos + 1][1] == 'I':
                      break;
                    i_pos += 1
                    labeled_entity.append(classified_labels[i_pos])
            # if our two entities are the same, then it means the
            # entity was correctly labeled from end to end
            if gold_entity == labeled_entity:
                true_pos += 1
            # if not then it is considered a false negative since
            # the gold labels had an entity that didn't match up
            # with the labeled entity
            else:
                false_neg += 1
    # do the math and return the value
    if true_pos + false_neg == 0:
        return 0.0
    return true_pos / (true_pos + false_neg)


def f1(gold_labels, classified_labels):
    """
  TODO: implement
  2pr/(p + r)
  params:
    gold_labels - a list of tuples in the format [(token, label), (token, label)...]
    classified_labels - a list of tuples in the format [(token, label), (token, label)...]
  return:
    float value of f1 at the entity level
  """
    # since we have implemented precision and recall and that
    # is all we need for the f1 math, just call those and
    # plug the numbers we got into an equation
    p = precision(gold_labels, classified_labels)
    r = recall(gold_labels, classified_labels)
    if p + r == 0:
      return 0
    return (2 * p * r) / (p + r)


def pretty_print_table(data, list_of_dicts):
    """
  Pretty-prints probability and backpointer lists of dicts as nice tables.
  Truncates column header words after 10 characters.
  params:
    data - list of words to serve as column headers
    list_of_dicts - list of dicts with len(data) dicts and the same set of
      keys inside each dict
  return: None
  """
    # ensure that each dict has the same set of keys
    keys = None
    for d in list_of_dicts:
        if keys is None:
            keys = d.keys()
        else:
            if d.keys() != keys:
                print("Error! not all dicts have the same keys!")
                return
    header = "\t" + "\t".join(['{:11.10s}'] * len(data))
    header = header.format(*data)
    rows = []
    for k in keys:
        r = k + "\t"
        for d in list_of_dicts:
            if type(d[k]) is float:
                r += '{:.9f}'.format(d[k]) + "\t"
            else:
                r += '{:10.9s}'.format(str(d[k])) + "\t"
        rows.append(r)
    print(header)
    for row in rows:
        print(row)


"""
Implement any other non-required functions here
"""

"""
Implement the following class
"""


class NamedEntityRecognitionHMM:

    def __init__(self):
        # TODO: implment as needed
        # pi[state] = count of sentences that began with that state
        self.pi = Counter()
        # transitions[prev_state][state] = count
        self.transitions = {}
        # emissions[state][word] = count
        self.emissions = {}
        self.states = set()
        self.vocab = set()
        self.tag_count = {}

    def train(self, examples):
        """
    Trains this model based on the given input data
    params: examples - a list of lists of (token, label) tuples
    Looks like [[(token, label), ...], ...]
    return: None
    """
        # iterate over our sentences in the examples
        for sentence in examples:
            # some testing prints
            # print('---------------------------')
            # print(sentence)
            # print('---------------------------')
            # get every tuple in the sentence
            for i in range(len(sentence)):
                # seperate the word and the state
                word = sentence[i][0]
                state = sentence[i][1]
                # add our word and state to our set of all words and states
                self.vocab.add(word)
                self.states.add(state)
                # if we are at the first word in the sentence need to
                # increment the number of times this tag appeared first in a sentence
                if i == 0:
                    self.pi[state] += 1
                # else we need to increment the number of times the
                # current tag was preceeded by the tag before it
                else:
                    if sentence[i - 1][1] not in self.transitions:
                        self.transitions[sentence[i - 1][1]] = Counter()
                    self.transitions[sentence[i - 1][1]][state] += 1
                # now we increment the number of times the word had this tag
                if state not in self.emissions:
                    self.emissions[state] = Counter()
                self.emissions[state][word] += 1
        # print(self.emissions)
        # print(self.transitions)
        # print(self.pi)
        # print('---------------------------')

        # now we store the counts we will need since during our iterations
        # the counts will change
        # this stores how many sentences we have
        # count(sentences)
        pi_val = sum(self.pi.values())
        # now we are going to get the counts of the tags
        # count(t_i)
        # we are using emissions because each tag occurs in it unlike
        # in transitions where the last tag is lost kind of
        for state in self.emissions.keys():
            # print(state, sum(self.emissions[state].values()))
            self.tag_count[state] = sum(self.emissions[state].values())
        # print('---------------------------')
        # now we do the probability of a sentence starting with each tag
        # count(t_i) / count(sentences)
        for state in self.pi:
            self.pi[state] /= pi_val
        # now we will calculate the probabilites that each tag proceeds the next tag
        # ie p(t_i | t_i-1) = count(t_i-1, t_i) / count(t_i-1)
        for prev_state in self.transitions:
            for state in self.transitions[prev_state]:
                # print(prev_state, state, self.transitions[prev_state][state])
                # print(prev_state, tag_count[prev_state])
                self.transitions[prev_state][state] /= self.tag_count[prev_state]
                # print(self.transitions[prev_state][state])
        # print('---------------------------')
        # and the probability of a word having the tag with laplace smoothing
        # p(w_i | t_i) = count(t_i, w_i) / count(t_i)
        for state in self.emissions:
            for word in self.emissions[state]:
                # print(state, word, self.emissions[state][word])
                # print(state, tag_count[state])
                self.emissions[state][word] = (self.emissions[state][word] + 1) / (
                            self.tag_count[state] + len(self.vocab))
                # print(self.emissions[state][word])
        # print('---------------------------')
        # print(self.emissions)
        # print(self.transitions)
        # print(self.pi)
        # print('---------------------------')
        # print(len(self.vocab))
        # print(len(self.states))
        # print('---------------------------')

    def generate_probabilities(self, data, laplace=True, beam=0):
        """
    params: data - a list of tokens
    return: two lists of dictionaries --
      - first a list of dictionaries of states to probabilities, 
      one dictionary per word in the test data that represents the
      probability of being at that state for that word
      - second a list of dictionaries of states to states, 
      one dictionary per word in the test data that represents the 
      backpointers for which previous state led to the best probability
      for the current state
    """
        # list of dictionaries
        # each dictionary is the probability of states for the word
        dics_probs = []
        # list of dictionaries
        # each dictionary has the backpointers for every state for the word
        dics_backptrs = []
        # iterate through every token in our data
        for i in range(len(data)):
            # get the token and store it
            word = data[i]
            # these are our dictionaries for this token
            probs = {}
            backptrs = {}
            # if we are at the beginning of a sentence
            if i == 0:
                # calculate the probability of starting with each tag and store None as the backpointer
                for tag in self.states:
                    if self.emissions[tag][word] == 0:
                        probs[tag] = 1 / (self.tag_count[tag] + len(self.vocab))
                        probs[tag] = probs[tag] * self.pi[tag]
                    else:
                        probs[tag] = self.emissions[tag][word] * self.pi[tag]
                    backptrs[tag] = None
            else:
                # calculate the probability for every tag
                for tag in self.states:
                    # store the maximum probability we get
                    max = float(-1)
                    # iterate through every tag
                    for prev_tag in self.states:
                        # get the probability of arriving at our tag given the previous tag and the word
                        # P(t_i) = P(t_i-1) * P(w_i | t_i) * P(t_i | t_i-1)
                        if self.emissions[tag][word] == 0:
                            emission = 1 / (self.tag_count[tag] + len(self.vocab))
                        else:
                            emission = self.emissions[tag][word]
                        current = float(dics_probs[i - 1][prev_tag] * emission * self.transitions[prev_tag][tag])
                        # if it is greater than the max then save it as our max and set the backpointer
                        # to the tag we came from
                        if current > max:
                            max = current
                            probs[tag] = current
                            backptrs[tag] = prev_tag
            # append the dictionaries we got to our list
            # print('---------------------------')
            # print(probs)
            # print(backptrs)
            # print('---------------------------')
            dics_probs.append(probs)
            dics_backptrs.append(backptrs)
        # pretty_print_table(dics_probs)
        # pretty_print_table(dics_backptrs)
        return dics_probs, dics_backptrs

    def __str__(self):
        return "HMM"


"""
Implement the following class
"""


class NamedEntityRecognitionMEMM:
    def __init__(self):
        # implement as needed
        self.num_feats = 102
        self.states = set(['I', 'O', 'B'])
        self.weights = {}
        for state in self.states:
            self.weights[state] = np.random.uniform(low=0.01, high=1.01, size=self.num_feats)
        self.epochs = 100
        self.loss = 0.1
        self.model = None

    def train(self, examples):
        """
        Trains this model based on the given input data
        params: examples - a list of lists of (token, label) tuples
        return: None
        """
        print(examples)
        # first we will do gensim to get word embeddings
        tokens = []
        for example in examples:
            for tuple in example:
                tokens.append([tuple[0]])
        self.model = Word2Vec(tokens, min_count=1, size=100).wv
        # shuffle the examples so that they are gone through 'randomly'
        #print(examples)
        random.shuffle(examples)
        #print(examples)
        # iterate through our examples
        for j in range(len(examples)):
            # the stored label for the previous token
            prev_label = None
            prev_word = None
            # iterate through our tokens for the example
            for i in range(len(examples[j])):
                # store our token and its label
                token = examples[j][i][0]
                y = examples[j][i][1]
                # get the features for our current token
                next_word = None
                if i <= (len(examples)-1):
                    next_word = examples[j][i+1][0]
                features = self.featurize(prev_label, prev_word, token, next_word)
                # set our previous label to our current since
                # we are done featurizing and need to store it for
                # the next iteration
                prev_label = y
                # a dictionary that will store our z values
                z = {}
                # calculate our z value for every state for
                # the example we are on
                # z(state) = features * weights
                # z[state] = np.dot(features, weights[state])
                for state in self.states:
                    z[state] = np.dot(features, self.weights[state])
                # store our max
                max = -1
                # store our y_hat
                y_hat = None
                # store our probabilities
                prob = {}
                # this runs softmax on our z's
                # y_hat = softmax(z)
                for state in self.states:
                    # softmax = p(state) = e^z[state] / (sum[e^z for all z's)
                    # making sure this works the way I want it to, should
                    # be three values
                    #print(np.array(list(z.values())))
                    #print(np.exp(np.array(list(z.values()))))
                    prob[state] = np.exp(z[state]) / sum(np.exp(np.array(list(z.values()))))
                    # if our current prob is greater than the others then it is our boy
                    if prob[state] > max:
                        # save the new prob as the max
                        max = prob[state]
                        # save the state as our prediction y_hat
                        y_hat = state
                # this will hold our gradients for all the states
                gradients = {}
                for state in self.states:
                    # gradient[state] = ((y_hat == state) - prob[state]) * features
                    gradients[state] = ((y_hat == state) - prob[state]) * features
                    # weights[state] -= loss * gradients
                    self.weights[state] -= self.loss * gradients[state]
        print(self.weights)

    def featurize(self, prev_tag, prev_word, token, next_word):
        """
        CHOOSE YOUR OWN PARAMS FOR THIS FUNCTION
        CHOOSE YOUR OWN RETURN VALUE FOR THIS FUNCTION
        """
        features = np.array([])
        # one of our features will be the previous tag
        if None or prev_tag == 'O':
            features = np.append(features, [0])
        else:
            features = np.append(features, [1])
        # another set of features will be our word shape

        # another feature will be our pos tag

        # another feature will be our word embeddings
        features = np.append(features, self.model[token])
        # and the final feature will be bias
        features = np.append(features, [1])
        return features

    def generate_probabilities(self, data):
        # returns dics of probabilites and backptrs for each word
        prev_tag = None
        prev_word = None
        back_ptr = None
        dics_probs = []
        dics_ptrs = []
        for i in range(len(data)):
            next_word = None
            word = data[i]
            if i <= (len(data) - 2):
                next_word = data[i+1]
            features = self.featurize(prev_tag, prev_word, word, next_word)
            y_hat = {}
            max = -1
            max_state = None
            for state in self.states:
                y_hat[state] = np.dot(self.weights[state], features)
            denom = sum(np.exp(np.array(list(y_hat.values()))))
            for state in self.states:
                y_hat[state] = np.exp(y_hat[state]) / denom
                if y_hat[state] > max:
                    max = y_hat[state]
                    max_state = state
            ptrs = {'I': back_ptr, 'O': back_ptr, 'B': back_ptr}
            dics_probs.append(y_hat)
            dics_ptrs.append(ptrs)
            back_ptr = max_state
        return dics_probs, dics_ptrs

    def __str__(self):
        return "MEMM"


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage:", "python hw5_ner.py training-file.txt testing-file.txt")
        sys.exit(1)

    training = sys.argv[1]
    testing = sys.argv[2]
    training_examples = generate_tuples_from_file(training)
    testing_examples = generate_tuples_from_file(testing)

    # instantiate each class, train it on the training data, and
    # evaluate it on the testing data
    MEMM = NamedEntityRecognitionHMM()
    MEMM.train(training_examples)
    labels, ptrs = MEMM.generate_probabilities(testing_examples)

