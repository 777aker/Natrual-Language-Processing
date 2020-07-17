import nltk
enc = 'utf-8'
file = open('essay1.txt', 'r', encoding=enc)
text1 = file.read()
file.close()
file = open('essay2.txt', 'r', encoding=enc)
text2 = file.read()
file.close()
file = open('essay3.txt', 'r', encoding=enc)
text3 = file.read()
file.close()
file = open('essay4.txt', 'r', encoding=enc)
text4 = file.read()
file.close()
file = open('essay5.txt', 'r', encoding=enc)
text5 = file.read()
file.close()

lengths = []

tokens = nltk.word_tokenize(text1)
uniques = set(tokens)
lengths.append(len(uniques))
tokens = nltk.word_tokenize(text2)
uniques = set(tokens)
lengths.append(len(uniques))
tokens = nltk.word_tokenize(text3)
uniques = set(tokens)
lengths.append(len(uniques))
tokens = nltk.word_tokenize(text4)
uniques = set(tokens)
lengths.append(len(uniques))
tokens = nltk.word_tokenize(text5)
uniques = set(tokens)
lengths.append(len(uniques))

sum = 0
number = 0
for f in lengths:
    for j in lengths:
        if not (f == j):
            number += 1
            sum += abs(f - j)

for i in lengths:
    print(i)
sum /= number
print(sum)

