import numpy
import pandas as pd
import math
import pickle

def countWords(count, sentence, cls):
	words = sentence.split()
	for word in words:
		count[cls][word] = count[cls].get(word,0) + 1

def readFiles(reviewFileName, classLabelFileName, count, nr):
	reviewFile = open(reviewFileName, 'r')
	classLabelFile = open(classLabelFileName, 'r')
	totalRecords = 0
	while True:
		review = reviewFile.readline()
		if review == '':
			break
		label = classLabelFile.readline()
		review = review.strip()
		label = label.strip()
		countWords(count, review, label)
		nr[label] = nr.get(label,0) + 1
		totalRecords += 1
	return totalRecords

def classProbabilities(numberOfRecords, reviewCount):
	probability = {}
	for key in numberOfRecords:
		probability[key] = numberOfRecords[key]/reviewCount
	return probability

def getWordsPerClass(countOfWords):
	wordsPerClass = dict()
	for key in countOfWords:
		wordsPerClass[key] = sum(countOfWords[key].values())
	return wordsPerClass

def probabilityWordGivenClass(word, classLabel, count, wordsPerClass, numberOfwords):
	numerator = count[classLabel].get(word,0) + 1
	denominator = wordsPerClass[classLabel] + numberOfwords
	return numerator/denominator

def getNumberOfWordsInVocab(countOfWords):
	vocab = set()
	for key in countOfWords:
		vocab = vocab.union(set(countOfWords[key].keys()))
	return len(vocab)

def createVocabulary():
	reviewFileName = './imdbDataset/imdb_train_text.txt'
	reviewFile = open(reviewFileName, 'r')
	reviews = reviewFile.readlines()
	words = []
	for review in reviews:
		words += review.split()
	vocab = sorted(list(set(words)))
	reviewFile.close()
	return vocab

def getIndexDict(vocab):
	n = len(vocab)
	index = {}
	for i in range(n):
		index[vocab[i]] = i
	return index

# This is not working because of memory error
def createFeatureVectors(vocab, index):
	#reviewFileName = './imdbDataset/imdb_train_text.txt'
	#classLabelFileName = './imdbDataset/imdb_train_labels.txt'
	reviewFileName = './imdbDataset/training.txt'
	classLabelFileName = './imdbDataset/imdb_train_labels.txt'
	reviewFile = open(reviewFileName, 'r')
	classLabelFile = open(classLabelFileName, 'r')
	reviews = reviewFile.readlines()
	x = numpy.zeros((len(vocab), len(reviews)))
	for i in range(len(reviews)):
		words = reviews[i].split()
		for word in words:
			x[index[word]][i] += 1

	df = pd.DataFrame(x, columns=vocab)
	df.to_csv('featureVectors.csv')
	reviewFile.close()
	classLabelFile.close()
	return x

def test(testDataFileName, testLabelFileName, classLabels, countOfWords, numberOfRecords, reviewCount, phi_k, wordsPerClass, totalDistinctWords):
	confusionMatrix = numpy.zeros((8,8), dtype=int)
	testDataFile = open(testDataFileName, 'r')
	testLabelFile = open(testLabelFileName, 'r')
	testData = testDataFile.readlines()
	testLabel = testLabelFile.readlines()
	testDataFile.close()
	testLabelFile.close()
	correct = 0
	total = 0
	for i,review in enumerate(testData):
		words = review.split()
		prediction = {}
		for label in classLabels:
			prediction[label] = math.log(phi_k[label])
		for word in words:
			for label in classLabels:
				prediction[label] += math.log(probabilityWordGivenClass(word, label, countOfWords, wordsPerClass, totalDistinctWords))
		maxValue = prediction['1']
		maxLabel = '1'
		for label in classLabels:
			if prediction[label] > maxValue:
				maxLabel = label
				maxValue = prediction[label]
		if maxLabel == testLabel[i].strip():
			correct += 1
		confusionMatrix[classLabels.index(maxLabel)][classLabels.index(testLabel[i].strip())] += 1
		total += 1
	print('Number of correct predictions : ', correct)
	print('Total number of tests : ', total)
	print('Accuracy = {}'.format(correct*100/total))
	print('Confusion Matrix- ')
	print(confusionMatrix)
	print(sum(confusionMatrix[[0,1,2,3,4,5,6,7],[0,1,2,3,4,5,6,7]]))


'''vocab = createVocabulary()
index = getIndexDict(vocab)
feature =  createFeatureVectors(vocab, index)'''

classLabels = ['1','2','3','4','7','8','9','10']


# -------------------------------------------------------------------------------------------------------------- #

countOfWords = {}
numberOfRecords = {}
for label in classLabels:
	countOfWords[label] = {}

reviewCount = readFiles('./imdbDataset/imdb_train_text.txt', './imdbDataset/imdb_train_labels.txt', countOfWords, numberOfRecords)
phi_k = classProbabilities(numberOfRecords, reviewCount)
wordsPerClass = getWordsPerClass(countOfWords)
totalDistinctWords = getNumberOfWordsInVocab(countOfWords)
print('Testing on training data ')
test('./imdbDataset/imdb_train_text.txt','./imdbDataset/imdb_train_labels.txt', classLabels, countOfWords, numberOfRecords, reviewCount, phi_k, wordsPerClass, totalDistinctWords)
print('Testing on test data ')
test('./imdbDataset/imdb_test_text.txt','./imdbDataset/imdb_test_labels.txt', classLabels, countOfWords, numberOfRecords, reviewCount, phi_k, wordsPerClass, totalDistinctWords)

print(totalDistinctWords)
print(wordsPerClass)

params = dict()
params['reviewCount'] = reviewCount
params['countOfWords'] = countOfWords
params['numberOfRecords'] = numberOfRecords
params['phi_k'] = phi_k
params['wordsPerClass'] = wordsPerClass
params['totalDistinctWords'] = totalDistinctWords
model1_out = open('model1.pickle', 'wb')
pickle.dump(params, model1_out)
model1_out.close()
# --------------------------------------------------------------------------------------------------------------- #

countOfWords = {}
numberOfRecords = {}
for label in classLabels:
	countOfWords[label] = {}

reviewCount = readFiles('./imdbDataset/training.txt', './imdbDataset/imdb_train_labels.txt', countOfWords, numberOfRecords)
phi_k = classProbabilities(numberOfRecords, reviewCount)
wordsPerClass = getWordsPerClass(countOfWords)
totalDistinctWords = getNumberOfWordsInVocab(countOfWords)
test('./imdbDataset/test.txt','./imdbDataset/imdb_test_labels.txt', classLabels, countOfWords, numberOfRecords, reviewCount, phi_k, wordsPerClass, totalDistinctWords)
params = dict()
params['reviewCount'] = reviewCount
params['countOfWords'] = countOfWords
params['numberOfRecords'] = numberOfRecords
params['phi_k'] = phi_k
params['wordsPerClass'] = wordsPerClass
params['totalDistinctWords'] = totalDistinctWords
model2_out = open('model2.pickle', 'wb')
pickle.dump(params, model2_out)
model2_out.close()

# -------------------------------------------------------------------------------------------------------------- #

countOfWords = {}
numberOfRecords = {}
for label in classLabels:
	countOfWords[label] = {}

reviewCount = readFiles('./imdbDataset/lowerTrain.txt', './imdbDataset/imdb_train_labels.txt', countOfWords, numberOfRecords)
phi_k = classProbabilities(numberOfRecords, reviewCount)
wordsPerClass = getWordsPerClass(countOfWords)
totalDistinctWords = getNumberOfWordsInVocab(countOfWords)
test('./imdbDataset/lowerTest.txt','./imdbDataset/imdb_test_labels.txt', classLabels, countOfWords, numberOfRecords, reviewCount, phi_k, wordsPerClass, totalDistinctWords)