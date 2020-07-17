import unittest
import hw5_ner as hw5

# updated 3/5/2020 to fix ordering issues in tests


class TestNERHMMMiniTrain(unittest.TestCase):
    
    def setUp(self):
        #Sets the Training File Path
        # Feel free to edit to reflect where they are on your machine
        self.training_file ="minitrain.txt"
        self.train_tups = hw5.generate_tuples_from_file(self.training_file)

    def test_precisionFunctionAlone(self):
        #Tests the tuple generation from the sentences
        labeled = hw5.generate_tuples_from_file(self.training_file)
        labeled[0][3] = ("phosphatases", "O") # incorrect boundary (ends early)
        labeled[0][5] = ("5", "O") # incorrect boundary (begins late)
        labeled[0][6] = ("-", "B") # incorrect boundary (begins late)
        labeled[1][4] = ("the", "B") # should be O
        train_tups = [tup for sent in self.train_tups for tup in sent]
        labeled = [tup for sent in labeled for tup in sent]
        precision = hw5.precision(train_tups, labeled)
        # 1 correct / 4 guessed
        self.assertEqual(.25, precision)
        labeled = hw5.generate_tuples_from_file(self.training_file)
        labeled[1][4] = ("the", "B") # should be O
        labeled = [tup for sent in labeled for tup in sent]
        precision = hw5.precision(train_tups, labeled)
        # 3 correct / 4 guessed
        self.assertEqual(.75, precision)

    def test_recallFunctionAlone(self):
        #Tests the tuple generation from the sentences
        labeled = hw5.generate_tuples_from_file(self.training_file)
        labeled[0][3] = ("phosphatases", "O") # incorrect boundary (ends early)
        labeled[0][5] = ("5", "O") # incorrect boundary (begins late)
        labeled[0][6] = ("-", "B") # incorrect boundary (begins late)
        labeled[1][4] = ("the", "B") # should be O
        train_tups = [tup for sent in self.train_tups for tup in sent]
        labeled = [tup for sent in labeled for tup in sent]
        recall = hw5.recall(train_tups, labeled)
        # 1 correct / 3 possible
        self.assertAlmostEqual(.33333333, recall)
        labeled = hw5.generate_tuples_from_file(self.training_file)
        labeled[1][4] = ("the", "B") # should be O
        labeled = [tup for sent in labeled for tup in sent]
        recall = hw5.recall(train_tups, labeled)
        # 3 correct / 3 possible
        self.assertEqual(1, recall)

    def test_f1Function(self):
        #Tests the tuple generation from the sentences
        labeled = hw5.generate_tuples_from_file(self.training_file)
        labeled[0][3] = ("phosphatases", "O") # incorrect boundary (ends early)
        labeled[0][5] = ("5", "O") # incorrect boundary (begins late)
        labeled[0][6] = ("-", "B") # incorrect boundary (begins late)
        labeled[1][4] = ("the", "B") # should be O
        train_tups = [tup for sent in self.train_tups for tup in sent]
        labeled = [tup for sent in labeled for tup in sent]
        f1 = hw5.f1(train_tups, labeled)
        realf1 = (2 * .25 * .333333) / (.25 + .333333)
        self.assertAlmostEqual(realf1, f1, places=5)
        labeled = hw5.generate_tuples_from_file(self.training_file)
        labeled[1][4] = ("the", "B") # should be O
        labeled = [tup for sent in labeled for tup in sent]
        f1 = hw5.f1(train_tups, labeled)
        realf1 = (2 * .75 * 1) / (.75 + 1)
        self.assertAlmostEqual(realf1, f1, places=5)


    def test_hmmprobspointers(self):
        hmm = hw5.NamedEntityRecognitionHMM()
        hmm.train(self.train_tups)
        test = [w[0] for w in self.train_tups[0]]
        probs, pointers = hmm.generate_probabilities(test)
        # correct shape
        self.assertEqual(9, len(probs))
        self.assertEqual(9, len(pointers))
        for row in probs:
            self.assertEqual(3, len(row))
            self.assertTrue(type(row) is dict)
            self.assertTrue(type(row['O'] is float))

        for row in pointers:
            self.assertEqual(3, len(row))
            self.assertTrue(type(row) is dict)
            self.assertTrue(type(row['O'] is str))

        # now we'll test the actual values
        # pi = 0, .5, .5 (I, O, B)
        vocab = 13
        p_Comparison_I = (0 + 1) / (5 + vocab)
        p_Comparison_0 = (1 + 1) / (8 + vocab)
        p_Comparison_B = (0 + 1) / (3 + vocab) 
        # WARNING THIS CODE DOES NOT FULLY TEST THE PROBABILITIES, ONLY 
        # THE FIRST TWO COLUMNS
        column1 = {"I": 0 * (p_Comparison_I), "O": 0.5 * (p_Comparison_0), "B": 0.5 * (p_Comparison_B)}
        self.assertEqual(column1, probs[0])

        p_in_I = (0 + 1) / (5 + vocab)
        p_in_O = (3 + 1) / (8 + vocab)
        p_in_B = (0 + 1) / (3 + vocab) 

        p_O_O = 4 / 8
        p_O_I = 3 / 5
        p_O_B = 0 / 3

        p_I_O = 0 / 8
        p_I_I = 2 / 5
        p_I_B = 3 / 3

        p_B_O = 2 / 8
        p_B_I = 0 / 5
        p_B_B = 0 / 3
        # the "missing" 2 for the O states are because the sentences end with O,
        # so these are accounted for in the distribution of pi

        prev_I_and_I = p_in_I * p_I_I * column1["I"]
        prev_O_and_I = p_in_I * p_I_O * column1["O"]
        prev_B_and_I = p_in_I * p_I_B * column1["B"]

        prev_I_and_O = p_in_O * p_O_I * column1["I"]
        prev_O_and_O = p_in_O * p_O_O * column1["O"]
        prev_B_and_O = p_in_O * p_O_B * column1["B"]

        prev_I_and_B = p_in_B * p_B_I * column1["I"]
        prev_O_and_B = p_in_B * p_B_O * column1["O"]
        prev_B_and_B = p_in_B * p_B_B * column1["B"]

        I_val = max(prev_I_and_I, prev_O_and_I, prev_B_and_I)
        O_val = max(prev_I_and_O, prev_O_and_O, prev_B_and_O)
        B_val = max(prev_I_and_B, prev_O_and_B, prev_B_and_B)

        self.assertAlmostEqual(I_val, probs[1]["I"])
        self.assertAlmostEqual(O_val, probs[1]["O"])
        self.assertAlmostEqual(B_val, probs[1]["B"])

        # ensure that back pointers are correct
        point_answers = [{'O': None, 'I': None, 'B': None}, {'O': 'O', 'I': 'B', 'B': 'O'}, {'O': 'O', 'I': 'B', 'B': 'O'}, {'O': 'O', 'I': 'B', 'B': 'O'}, {'O': 'I', 'I': 'I', 'B': 'O'}, {'O': 'O', 'I': 'I', 'B': 'O'}, {'O': 'O', 'I': 'B', 'B': 'O'}, {'O': 'I', 'I': 'I', 'B': 'O'}, {'O': 'I', 'I': 'I', 'B': 'O'}]        
        self.assertEqual(point_answers, pointers)
        

    def test_decodehmm(self):
        hmm = hw5.NamedEntityRecognitionHMM()
        hmm.train(self.train_tups)
        test = [w[0] for w in self.train_tups[0]]
        probs, pointers = hmm.generate_probabilities(test)

        # find the actual labels
        labels = hw5.decode(test, probs, pointers)
        label_answers = [('Comparison', 'O'), ('in', 'O'), ('alkaline', 'B'), ('phosphatases', 'I'), ('in', 'O'), ('5', 'B'), ('-', 'I'), ('nucleotidase', 'I'), ('.', 'O')]
        self.assertEqual(label_answers, labels)


    def test_memmprobspointers(self):
        memm = hw5.NamedEntityRecognitionMEMM()
        memm.train(self.train_tups)#, iterations = 100)
        test = [w[0] for w in self.train_tups[0]]
        probs, pointers = memm.generate_probabilities(test)
        labeled = hw5.decode(test, probs, pointers)
        # print(labeled)
        # # correct shape
        self.assertEqual(len(test), len(probs))
        self.assertEqual(len(test), len(pointers))
        for row in probs:
            self.assertEqual(3, len(row))
            self.assertTrue(type(row) is dict)
            self.assertTrue(type(row['O'] is float))

        for row in pointers:
            self.assertEqual(3, len(row))
            self.assertTrue(type(row) is dict)
            self.assertTrue(type(row['O'] is str))        

        # ensure that back pointers are correct
        # you should be able to end up with these answers within 
        # 100 or fewer iterations of SGD
        point_answers = [{'B': None, 'I': None, 'O': None}, {'B': 'O', 'I': 'O', 'O': 'O'}, {'B': 'O', 'I': 'O', 'O': 'O'}, {'B': 'B', 'I': 'B', 'O': 'B'}, {'B': 'I', 'I': 'I', 'O': 'I'}, {'B': 'O', 'I': 'O', 'O': 'O'}, {'B': 'B', 'I': 'B', 'O': 'B'}, {'B': 'I', 'I': 'I', 'O': 'I'}, {'B': 'I', 'I': 'I', 'O': 'I'}]
        self.assertEqual(point_answers, pointers)

    def test_decodememm(self):
        memm = hw5.NamedEntityRecognitionMEMM()
        memm.train(self.train_tups)
        test = [w[0] for w in self.train_tups[0]]
        probs, pointers = memm.generate_probabilities(test)

        # find the actual labels
        labels = hw5.decode(test, probs, pointers)
        label_answers = [('Comparison', 'O'), ('in', 'O'), ('alkaline', 'B'), ('phosphatases', 'I'), ('in', 'O'), ('5', 'B'), ('-', 'I'), ('nucleotidase', 'I'), ('.', 'O')]
        self.assertEqual(label_answers, labels)

if __name__ == "__main__":
    print("Usage: python test_minitraining.py")
    unittest.main()

