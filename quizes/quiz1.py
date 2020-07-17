import nltk
enc = 'utf-8'
file = open('MobyDick.txt', 'r', encoding=enc)
text = file.read()
file.close()
tokens = nltk.word_tokenize(text)
print(len(tokens))
uniques = set(tokens)
print(len(uniques))