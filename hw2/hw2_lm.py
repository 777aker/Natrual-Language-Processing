import numpy as np
import matplotlib.pyplot as plt
import sys

class LanguageModel:

    # this is the class declaration function
    def __init__(self, n_gram, is_laplace_smoothing, backoff = None):
        # initialize all the variables used by the class as a whole and
        # store the values needed from the user that were passed
        self.n_gram = n_gram
        self.is_laplace_smoothing = is_laplace_smoothing
        self.grams = {}
        self.gramsminusone = {}
        self.corpus = {}
        self.corpussize = 0
        self.vocabsize = 0

    # this is the function that will take in text
    # files and count our ngrams and ngrams - one
    def train(self, training_file_path):
        # open the file, read the file, then close the file
        file = open(training_file_path, 'r')
        text = file.read()
        file.close()
        # split the file by lines since each line is a sentence
        sentences = text.split('\n')

        # loop through every sentence and add each word to a dict
        # of words and their counts
        for sentence in sentences:
            # split the sentence by words
            words = sentence.split(' ')
            if '' in words:
                words.remove('')
            # also going to store every word in a dict
            for word in words:
                if word in self.corpus:
                    self.corpus[word] += 1
                else:
                    self.corpus[word] = 1
        # a list of words to delete from the corpus once we
        # loop through it
        delete = []
        unknowns = 0
        # loop through the corpus and change words below a certain
        # threshold to <UNK>
        for word in self.corpus:
            # if the word occurs below the threshold then append
            # it to the list of words to delete from the corpus
            # and add how many times it occurred to <UNK>
            if self.corpus[word] <= 1:
                temp = self.corpus[word]
                delete.append(word)
                unknowns += temp
        # delete every word in the corpus in the delete list
        for word in delete:
            del self.corpus[word]
        # create the <UNK> in the dictionary
        if not unknowns == 0:
            if '<unk>' in self.corpus:
                self.corpus['<unk>'] += unknowns
            else:
                self.corpus['<unk>'] = unknowns
        # sum up the size of the corpus / how many words are in the corpus
        # and the vocab of the corpus
        for word in self.corpus:
            self.corpussize += self.corpus[word]
        self.vocabsize = len(self.corpus)

        # loop through every sentences
        for sentence in sentences:
            # split up the sentence by every word
            words = sentence.split(' ')
            if '' in words:
                words.remove('')
            # if a word is not in a corpus then change it to <UNK>
            for i in range(len(words)):
                if not words[i] in self.corpus:
                    words[i] = '<unk>'
            # loop over every word in the sentence and create our
            # ngrams by looking at our ngram size and adding that
            # many words after our current word to create ngrams
            for i in range(len(words)-(self.n_gram-1)):
                # a variable for storing the current ngram
                gram = ''
                # for the size of our ngram model, add words to our ngram
                for j in range(self.n_gram):
                    gram += words[i + j] + ' '
                # if the gram is not stored, then store it with
                # a count of one
                if not gram in self.grams:
                    self.grams[gram] = 1
                # if the gram is stored, then increment the gram's count
                else:
                    self.grams[gram] += 1
                # now we need the counts of our grams without the last word
                # so store ngrams - 1 in a dict called gramsminusone
                minusgram = ''
                # add words to our gram but stop one word short of our size
                for k in range(self.n_gram-1):
                    minusgram += words[i + k] + ' '
                # if it is not in the dict of shortened grams then add it
                if not minusgram in self.gramsminusone:
                    self.gramsminusone[minusgram] = 1
                # if it is in the dict then increment the count by one
                else:
                    self.gramsminusone[minusgram] += 1
        # prints for testing
        #print(self.grams)
        #print(self.gramsminusone)
        #print(self.corpus)
        #print(self.vocabsize)
        #print(self.corpussize)

    def generate(self, num_sentences):
        # the list of sentences generated
        sentences = []
        # until we reach the number of sentences wanted by the user
        for i in range(num_sentences):
            # this is the sentence we are currently generating
            sentence = []
            # append <s> to the beginning of the sentence depending
            # on the size of the ngram
            for n in range(self.n_gram-1):
                sentence.append('<s>')
            # if we have a unigram then we need to do something special
            # since it doesn't have the same pattern of <s> and </s> that
            # all other ngrams have
            if self.n_gram == 1:
                # append a single <s> at the beginning
                sentence.append('<s>')
                # while the sentence hasn't ended, append words randomly
                # selected from the probability distribution
                while not sentence[-1] == '</s>':
                    # store the keys and values for our random choice
                    keys = []
                    values = []
                    # for every word in the corpus get it's probabiltity of
                    # occuring
                    for k in self.corpus:
                        if not k == '<s>':
                            keys.append(k)
                            values.append(self.corpus[k]/(self.corpussize-self.corpus['<s>']))
                    # append a randomly chosen word based on the probabilities to
                    # the sentence
                    sentence.append(np.random.choice(a=keys, p=values))
            else:
                # how many </s> we need at the end of a sentence depends
                # on n for ngrams so calculate how many sentence ends we need
                endtokens = self.n_gram-1
                # while we haven't reached the number of sentence ends we need
                # for the ngram add words
                # note that it's ok to do this because the probability of an </s>
                # after another </s> will be 100 percent so we don't need to worry
                # about it randomly selecting something like <s> <s> hello </s> I </s>
                # this also means that as long as the element at say -3 is </s> all the ones
                # after that are </s> so we don't need to look at anything after -3
                while not sentence[-endtokens] == '</s>':
                    # get the ngram - 1 for calculating probabilities
                    i = self.n_gram - 1
                    gram = ''
                    # get the last n - 1 words from our sentence so we can
                    # select the next word based off of the last ones
                    while not i == 0:
                        gram += sentence[-i] + ' '
                        i -= 1
                    # store possible words and their probabilities
                    keys = []
                    values = []
                    # go through every possible word we could append
                    for word in self.corpus:
                        # if that combination exists in our training data
                        newgram = gram + word + ' '
                        if newgram in self.grams:
                            # then add that word to possible choices and get the
                            # probability of that word occurring
                            keys.append(word)
                            values.append(self.grams[newgram]/self.gramsminusone[gram])
                    # choose a random word to append to the sentence based
                    # on the calculated probabilities
                    sentence.append(np.random.choice(a=keys, p=values))
            # put our finished sentence into the sentences we have generated
            # in a nicely formatted way
            s = ''
            for word in sentence:
                s += word + ' '
            sentences.append(s)
        return sentences

    def score(self, sentence):
        sentence = sentence.lower()
        # split the sentence into each word
        words = sentence.split(' ')
        if '' in words:
            words.remove('')
        # store the probability
        probability = 0.0
        # convert all unknown words to <UNK>
        for i in range(len(words)):
            if not (words[i]) in self.corpus:
                words[i] = '<unk>'
        # if it is a unigram we don't use gramsminusone and grams so have
        # a seperate case for it and ngrams since all other ngrams will use
        # grams and gramsminusone for their probability
        if self.n_gram == 1:
            # if it is laplace smoothing then do probability based on laplace
            if self.is_laplace_smoothing:
                # take the probability of each word, log that probability,
                # then add that to the probability variable
                for word in words:
                    if word in self.corpus:
                        toplog = self.corpus[word] + 1
                        botlog = self.corpussize + self.vocabsize
                        log = float(toplog)/float(botlog)
                    else:
                        toplog = 1
                        botlog = self.corpussize + self.vocabsize
                        log = float(toplog)/float(botlog)
                    probability += np.log(log)
            # if not laplace
            else:
                # calculate the probability of each word and add the log
                # of that probability to our probability variable
                for word in words:
                    if word in self.corpus:
                        log = float(self.corpus[word])/float(self.corpussize)
                    else:
                        return 0
                    probability += np.log(log)
        # if it is an ngram that is not a unigram then
        else:
            # first convert all unknown words to <UNK>
            for i in range(len(words)):
                if not words[i] in self.corpus:
                    words[i] = '<unk>'
            # if laplace smoothing then do the probability this way
            if self.is_laplace_smoothing:
                # loop through the sentence
                for i in range(len(words)-(self.n_gram-1)):
                    # store the gram we are currently looking at
                    gram = ''
                    # creates the gram we are currently looking at
                    for f in range(self.n_gram):
                        gram += words[i + f] + ' '
                    # creates the current gram minus the last element
                    gramminus = ''
                    for f in range(self.n_gram-1):
                        gramminus += words[i + f] + ' '
                    # calculate the probability of this gram and add the
                    # log of that probability to our probability variable
                    # if the gram hasn't shown up in our data then don't reference the
                    # data and we have never seen it before, if it has then compute the
                    # probability as normal
                    if gram in self.grams:
                        toplog = self.grams[gram] + 1
                    else:
                        toplog = 1
                    if gramminus in self.gramsminusone:
                        botlog = self.gramsminusone[gramminus] + self.vocabsize
                    else:
                        botlog = self.vocabsize
                    log = float(toplog)/float(botlog)
                    # if the probability of any of our grams is 0 then just
                    # return 0 since that will be the overall probability
                    if log == 0:
                        return 0
                    probability += np.log(log)
            # if we aren't doing laplace smoothing
            else:
                # loop through the sentence
                for i in range(len(words)-(self.n_gram-1)):
                    # create the gram we are currently looking at
                    gram = ''
                    for f in range(self.n_gram):
                        gram += words[i + f] + ' '
                    # create the gram we are currently looking at minus
                    # the last element
                    gramminus = ''
                    for f in range(self.n_gram-1):
                        gramminus += words[i + f] + ' '
                    # calculate the probability of our current gram
                    # if the gram given isn't in our data then return 0 as it's probability
                    if ((gram in self.grams) and (gramminus in self.gramsminusone)):
                        log = float(self.grams[gram])/float(self.gramsminusone[gramminus])
                    else:
                        return 0
                    # if the probability of our current gram is 0 then return 0
                    if log == 0:
                        return 0
                    # take the log of our probability and add it to our probability
                    # variable
                    probability += np.log(log)
        # take the exp of our probability so we put it back in 0 to 1 range
        probability = np.exp(probability)
        # return the probability we calculated for the sentence
        return probability



if __name__ == "__main__":
    #print(sys.argv)
    if len(sys.argv) == 4:

        # create and train the unigram model
        unigram = LanguageModel(1, True)
        unigram.train(sys.argv[1])
        # generate random sentences based on the model and output
        # it to a text file
        generated = unigram.generate(100)
        file = open('hw2-unigram-generated.txt', 'w')
        for s in generated:
            file.write(s + '\n')
        file.close()
        # run the test files and output the probabilities to a file
        file = open(sys.argv[2])
        text = file.read()
        file.close()
        sentences = text.split('\n')
        file = open('hw2-unigram-out.txt', 'w')
        for s in sentences:
            # there are some blanks at the end of the files that
            # get interpreted, just removing them
            if not s == '':
                prob = unigram.score(s)
                file.write(str(prob) + '\n')
        file.close()
        # create the histogram and save histogram to a pdf
        # store the probabilities from the test data
        testprobs = []
        # read in the sentences and get the probability for each sentence
        file = open(sys.argv[2])
        text = file.read()
        file.close()
        sentences = text.split('\n')
        for s in sentences:
            prob = unigram.score(s)
            testprobs.append(prob)
        # store the probabilites from my sentences
        myprobs = []
        # read in the sentences and get the probability for each sentence
        file = open(sys.argv[3])
        text = file.read()
        file.close()
        sentences = text.split('\n')
        for s in sentences:
            prob = unigram.score(s)
            myprobs.append(prob)
        # create the figure for the data
        fig, ax = plt.subplots(figsize=(12,12))
        # find the minimum probability
        overall_min = min(min(testprobs), min(myprobs))
        min_exponent = np.floor(np.log10(np.abs(overall_min)))
        # create the bins for the histogram
        mybins = np.logspace(np.log10(10**min_exponent),np.log(1.0))
        # plut the data into a histogram
        plt.hist(testprobs, bins=mybins, fc='blue', alpha=0.5, label='Test Sentences')
        plt.hist(myprobs, bins=mybins, fc='red', alpha=0.5, label='My Sentences')
        # change the x scale to log
        plt.xscale('log')
        # show the legend
        ax.legend()
        # set labels of the histogram
        ax.set_ylabel('Frequency')
        ax.set_xlabel("Probability")
        ax.set_title('Frequencies of Probabilities for Unigram Model')
        # save the figure to a pdf
        plt.savefig('hw2-unigram-histogram.pdf')

        # create and train the bigram model
        bigram = LanguageModel(2, True)
        bigram.train(sys.argv[1])
        # generate random sentences based on the model and output
        # it to a text file
        generated = bigram.generate(100)
        file = open('hw2-bigram-generated.txt', 'w')
        for s in generated:
            file.write(s + '\n')
        file.close()
        # run the test files and output the probabilities to a file
        file = open(sys.argv[2])
        text = file.read()
        file.close()
        sentences = text.split('\n')
        file = open('hw2-bigram-out.txt', 'w')
        for s in sentences:
            # there are some blanks at the end of the files that
            # get interpreted, just removing them
            if not s == '':
                prob = bigram.score(s)
                file.write(str(prob) + '\n')
        file.close()
        # create the histogram and save histogram to a pdf
        # store the probabilities from the test data
        testprobs = []
        # read in the sentences and get the probability for each sentence
        file = open(sys.argv[2])
        text = file.read()
        file.close()
        sentences = text.split('\n')
        for s in sentences:
            prob = bigram.score(s)
            testprobs.append(prob)
        # store the probabilites from my sentences
        myprobs = []
        # read in the sentences and get the probability for each sentence
        file = open(sys.argv[3])
        text = file.read()
        file.close()
        sentences = text.split('\n')
        for s in sentences:
            prob = bigram.score(s)
            myprobs.append(prob)
        # create the figure for the data
        fig, ax = plt.subplots(figsize=(12, 12))
        # find the minimum probability
        overall_min = min(min(testprobs), min(myprobs))
        min_exponent = np.floor(np.log10(np.abs(overall_min)))
        # create the bins for the histogram
        mybins = np.logspace(np.log10(10 ** min_exponent), np.log(1.0))
        # plut the data into a histogram
        plt.hist(testprobs, bins=mybins, fc='blue', alpha=0.5, label='Test Sentences')
        plt.hist(myprobs, bins=mybins, fc='red', alpha=0.5, label='My Sentences')
        # change the x scale to log
        plt.xscale('log')
        # show the legend
        ax.legend()
        # set labels of the histogram
        ax.set_ylabel('Frequency')
        ax.set_xlabel("Probability")
        ax.set_title('Frequencies of Probabilities for Bigram Model')
        # save the figure to a pdf
        plt.savefig('hw2-bigram-histogram.pdf')

    # my personal testing
    else:

        testing = LanguageModel(1, False)
        testing.train('iamsam.txt')
        testing.score('helele')

        testing = LanguageModel(1, True)
        testing.train('iamsam.txt')
        testing.score('helele')

        testing = LanguageModel(2, False)
        testing.train('iamsam.txt')
        testing.score('helele')

        testing = LanguageModel(2, True)
        testing.train('iamsam.txt')
        testing.score('helele')

        '''
        print("1-gram -----------------------")
        testing = LanguageModel(1, False)
        testing.train('berp-training.txt')
        #sentences = testing.generate(5)
        #for s in sentences:
        #    print(s)
        print("2-gram -----------------------")
        testing = LanguageModel(2, False)
        testing.train('berp-training.txt')
        #sentences = testing.generate(5)
        #for s in sentences:
        #    print(s)
        '''
        '''
        print("3-gram -----------------------")
        testing = LanguageModel(3, False)
        testing.train('berp-training - trigram.txt')
        sentences = testing.generate(5)
        for s in sentences:
            print(s)
        print("4-gram -----------------------")
        testing = LanguageModel(4, False)
        testing.train('berp-training - 4-gram.txt')
        sentences = testing.generate(5)
        for s in sentences:
            print(s)
        '''