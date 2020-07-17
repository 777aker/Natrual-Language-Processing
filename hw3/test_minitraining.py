import unittest
import hw3_sentiment as hw3


class TestSentimentAnalysisBaselineMiniTrain(unittest.TestCase):
    
    def setUp(self):
        #Sets the Training File Path
        # Feel free to edit to reflect where they are on your machine
        self.trainingFilePath="training_files/minitrain.txt"
        self.devFilePath="training_files/minidev.txt"

    def test_GenerateTuplesFromTrainingFile(self):
        #Tests the tuple generation from the sentences
        sa = hw3.SentimentAnalysis()
        examples = hw3.generate_tuples_from_file(self.trainingFilePath)
        actualExamples = [('ID-2', 'The hotel was not liked by me', '0'), ('ID-3', 'I loved the hotel', '1'), ('ID-1', 'The hotel was great', '1'), ('ID-4', 'I hated the hotel', '0')]
        print("Tuple generation test")
        self.assertListEqual(sorted(examples), sorted(actualExamples))
        print("Done")
        
    def test_ScorePositiveExample(self):
        #Tests the Probability Distribution of each class for a positive example
        print("tests probability distribution for positive example")
        sa = hw3.SentimentAnalysis()
        examples = hw3.generate_tuples_from_file(self.trainingFilePath)
        #Trains the Naive Bayes Classifier based on the tuples from the training data
        sa.train(examples)
        #Returns a probability distribution of each class for the given test sentence
        score=sa.score("I loved the hotel")
        #P(C|text)=P(I|C)*P(loved|C)*P(the|C)*P(hotel|C)*P(C),where C is either 0 or 1(Classifier)
        pos = ((1+1)/(8+12))*((1+1)/(8+12))*((1+1)/(8+12))*((2+1)/(8+12))*(2/4)
        neg = ((1+1)/(11+12))*((0+1)/(11+12))*((1+1)/(11+12))*((2+1)/(11+12))*(2/4)
        actualScoreDistribution={'1': pos, '0': neg}
        self.assertAlmostEqual(score['0'], actualScoreDistribution['0'], places=5)
        self.assertAlmostEqual(score['1'], actualScoreDistribution['1'], places=5)
        print("Done")
        

    def test_ScorePositiveExampleWithUnkowns(self):
        #Tests the Probability Distribution of each class for a positive example
        print("tests probability disctribution with unknown for positive example")
        sa = hw3.SentimentAnalysis()
        examples = hw3.generate_tuples_from_file(self.trainingFilePath)
        #Trains the Naive Bayes Classifier based on the tuples from the training data
        sa.train(examples)
        #Returns a probability distribution of each class for the given test sentence
        score=sa.score("I loved the hotel a lot")
        #P(C|text)=P(I|C)*P(loved|C)*P(the|C)*P(hotel|C)*P(a|C)*P(lot|C)*P(C),where C is either 0 or 1(Classifier)
        pos = ((1+1)/(8+12))*((1+1)/(8+12))*((1+1)/(8+12))*((2+1)/(8+12))*(2/4)
        neg = ((1+1)/(11+12))*((0+1)/(11+12))*((1+1)/(11+12))*((2+1)/(11+12))*(2/4)
        actualScoreDistribution={'1': pos, '0': neg}
        self.assertAlmostEqual(score['0'], actualScoreDistribution['0'], places=5)
        self.assertAlmostEqual(score['1'], actualScoreDistribution['1'], places=5)
        print("Done")
        
    def test_ClassifyForPositiveExample(self):
        #Tests the label classified  for the positive test sentence
        print("tests the label for positive test sentence")
        sa = hw3.SentimentAnalysis()
        examples = hw3.generate_tuples_from_file(self.trainingFilePath)
        sa.train(examples)
        #Classifies the test sentence based on the probability distribution of each class
        label=sa.classify("I loved the hotel a lot")
        actualLabel='1'
        self.assertEqual(actualLabel,label)
        print("Done")
        
    def test_ScoreForNegativeExample(self):
        #Tests the Probability Distribution of each class for a negative example
        print("tests probability distribution of each class for negative example")
        sa = hw3.SentimentAnalysis()
        examples = hw3.generate_tuples_from_file(self.trainingFilePath)
        sa.train(examples)
        score=sa.score("I hated the hotel")
         #P(C|text)=P(I|C)*P(hated|C)*P(the|C)*P(hotel|C)*P(C),where C is either 0 or 1(Classifier)
        pos = ((1+1)/(8+12))*((0+1)/(8+12))*((1+1)/(8+12))*((2+1)/(8+12))*(2/4)
        neg = ((1+1)/(11+12))*((1+1)/(11+12))*((1+1)/(11+12))*((2+1)/(11+12))*(2/4)
        actualScoreDistribution={'1': pos, '0': neg}
        self.assertAlmostEqual(score['0'], actualScoreDistribution['0'], places=5)
        self.assertAlmostEqual(score['1'], actualScoreDistribution['1'], places=5)
        print("Done")
        
        
    def test_ClassifyForNegativeExample(self):
        #Tests the label classified  for the negative test sentence
        print("tests label classified for negative test sentence")
        sa = hw3.SentimentAnalysis()
        examples = hw3.generate_tuples_from_file(self.trainingFilePath)
        sa.train(examples)
        label=sa.classify("I hated the hotel")
        actualLabel='0'
        self.assertEqual(actualLabel,label)
        print("Done")

    def test_precision(self):
        print("tests precision")
        gold = [1, 1, 1, 0, 0]
        gold = [str(b) for b in gold]
        classified = [1, 0, 0, 0, 1]
        classified = [str(b) for b in classified]
        self.assertEqual((1 / 2), hw3.precision(gold, classified))
        print("Done")

    def test_recall(self):
        print("tests recall")
        gold = [1, 1, 1, 0, 0]
        gold = [str(b) for b in gold]
        classified = [1, 0, 0, 0, 1]
        classified = [str(b) for b in classified]
        self.assertEqual((1 / 3), hw3.recall(gold, classified))
        print("Done")

    # feel free to write your own test for your f1 score c


if __name__ == "__main__":
    print("Usage: python test_minitraining.py")
    unittest.main()

