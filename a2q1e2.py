import numpy
import pandas as pd
import math
import pickle

def wordVocabulary(reviews):
	vocab = {}
	for review in reviews:
		words = review.split()
		for word in words:
			vocab[word] = vocab.get(word, 0) + 1
	for k in list(vocab.keys()):
		if vocab[k] < 7:	#4
			del vocab[k]
	return vocab

def bigramVocabulary(reviews):
	vocab = {}
	for review in reviews:
		words = review.split()
		for i in range(len(words)-1):
			bigram = ' '.join(words[i:i+2])
			vocab[bigram] = vocab.get(bigram, 0) + 1
	for k in list(vocab.keys()):
		if vocab[k] < 6:	#3
			del vocab[k]
	return vocab

def readFiles(reviewFileName, classLabelFileName, countWords, countBigrams, nr):
	reviewFile = open(reviewFileName, 'r')
	classLabelFile = open(classLabelFileName, 'r')
	reviews = reviewFile.readlines()
	labels = classLabelFile.readlines()
	wordsVocab = wordVocabulary(reviews)
	bigramVocab = bigramVocabulary(reviews)
	totalRecords = 0
	for i,review in enumerate(reviews):
		label = labels[i].strip()
		review = review.strip()
		words = review.split()
		for i in range(len(words)-1):
			bigram = ' '.join(words[i:i+2])
			if bigram in bigramVocab:
				countBigrams[label][bigram] = countBigrams[label].get(bigram,0) + 1
		for word in words:
			if word in wordsVocab:
				countWords[label][word] = countWords[label].get(word,0) + 1
		nr[label] = nr.get(label,0) + 1
		totalRecords += 1
	return totalRecords

def classProbabilities(numberOfRecords, reviewCount):
	probability = {}
	for key in numberOfRecords:
		probability[key] = numberOfRecords[key]/reviewCount
	return probability

def getWordsPerClass(countOfWords, countOfBigrams):
	wordsPerClass = dict()
	for key in countOfWords:
		wordsPerClass[key] = sum(countOfWords[key].values())
	return wordsPerClass

def probabilityWordGivenClass(word, classLabel, countWords, wordsPerClass, numberOfwords):
	numerator = countWords[classLabel].get(word,0) + 1
	denominator = wordsPerClass[classLabel] + numberOfwords
	return numerator/denominator

def probabilityBigramGivenClass(bigram, word, classLabel, countBigrams, countWords, numberOfwords):
	numerator = countBigrams[classLabel].get(bigram,0) + 1
	denominator = countWords[classLabel].get(word,0) + numberOfwords
	return numerator/denominator

def getNumberOfWordsInVocab(countOfWords, countOfBigrams):
	vocab = set()
	for key in countOfWords:
		vocab = vocab.union(set(countOfWords[key].keys()))
	return len(vocab)

def test(testDataFileName, testLabelFileName, classLabels, countOfWords, 
	countOfBigrams, numberOfRecords, reviewCount, phi_k, wordsPerClass, totalDistinctWords):
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
		for j in range(len(words)-1):
			bigram = ' '.join(words[j:j+2])
			for label in classLabels:
				prediction[label] += math.log(probabilityBigramGivenClass(bigram, words[j], label, countOfBigrams, countOfWords, totalDistinctWords))
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
countOfBigrams = {}
numberOfRecords = {}
for label in classLabels:
	countOfWords[label] = {}
	countOfBigrams[label] = {}

reviewCount = readFiles('./imdbDataset/training.txt', './imdbDataset/imdb_train_labels.txt', countOfWords, countOfBigrams, numberOfRecords)
phi_k = classProbabilities(numberOfRecords, reviewCount)
wordsPerClass = getWordsPerClass(countOfWords, countOfBigrams)
totalDistinctWords = getNumberOfWordsInVocab(countOfWords, countOfBigrams)
test('./imdbDataset/test.txt','./imdbDataset/imdb_test_labels.txt', classLabels, countOfWords, countOfBigrams, numberOfRecords, reviewCount, phi_k, wordsPerClass, totalDistinctWords)
test('./imdbDataset/training.txt','./imdbDataset/imdb_train_labels.txt', classLabels, countOfWords, countOfBigrams, numberOfRecords, reviewCount, phi_k, wordsPerClass, totalDistinctWords)

params = dict()
params['reviewCount'] = reviewCount
params['countOfWords'] = countOfWords
params['numberOfRecords'] = numberOfRecords  
params['phi_k'] = phi_k
params['wordsPerClass'] = wordsPerClass
params['totalDistinctWords'] = totalDistinctWords
params['countOfBigrams'] = countOfBigrams

model1_out = open('bestModel.pickle', 'wb')
pickle.dump(params, model1_out)
model1_out.close()
# --------------------------------------------------------------------------------------------------------------- #