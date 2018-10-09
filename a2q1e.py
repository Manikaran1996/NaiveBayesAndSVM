import numpy
import pandas as pd
import math

def getBigramsCount(reviews, threshold=5):
	bigramCount = dict()
	for review in reviews:
		wordsInReview = review.split()
		for j in range(len(wordsInReview)):
			bigramCount[' '.join(wordsInReview[j:j+2])] = bigramCount.get(' '.join(wordsInReview[j:j+2]), 0) + 1
	keys = bigramCount.keys()
	for k in list(keys):
		if bigramCount[k] < threshold:
			del bigramCount[k]
	return bigramCount

def countBigram(count, sentence, bigramCount, cls):
	wordsInReview = sentence.split()
	for j in range(len(wordsInReview)-1):
		bigram = ' '.join(wordsInReview[j:j+2])
		if bigram in bigramCount:
			count[cls][bigram] = count[cls].get(bigram,0) + 1

def countWords(countOfWords, sentence, cls):
	wordsInReview = sentence.split()
	for word in wordsInReview:
		countOfWords[cls][word] = countOfWords[cls].get(word, 0) + 1

def getNumberOfWordsInVocab(countOfWords):
	vocab = set()
	for key in countOfWords:
		vocab = vocab.union(set(countOfWords[key].keys()))
	return len(vocab)

def readFiles(reviewFileName, classLabelFileName, count, countOfWords, nr):
	reviewFile = open(reviewFileName, 'r')
	classLabelFile = open(classLabelFileName, 'r')
	totalRecords = 0
	reviews = reviewFile.readlines()
	labels = classLabelFile.readlines()
	bigramCount = getBigramsCount(reviews, 1)
	print("Length of bigramCount : ", len(bigramCount))
	print(list(bigramCount.keys())[:50])
	for (review, label) in zip(reviews, labels):
		review = review.strip()
		label = label.strip()
		countBigram(count, review, bigramCount, label)
		countWords(countOfWords, review, label)
		nr[label] = nr.get(label,0) + 1
		totalRecords += 1
	return totalRecords, bigramCount

def classProbabilities(numberOfRecords, reviewCount):
	probability = {}
	for key in numberOfRecords:
		probability[key] = numberOfRecords[key]/reviewCount
	return probability

def getBigramsPerClass(countOfBigrams):
	bigramsPerClass = dict()
	for key in countOfBigrams:
		bigramsPerClass[key] = sum(countOfBigrams[key].values())
	return bigramsPerClass

#38.44
def probabilityBigramGivenClass(bigram, word, classLabel, count, countOfWords, numberOfWords):
	numerator = count[classLabel].get(bigram,0) + 1
	denominator = countOfWords[classLabel].get(word, 0) + numberOfWords
	return numerator/denominator

def test(testDataFileName, testLabelFileName, classLabels, countOfBigrams, countOfWords, numberOfRecords, reviewCount, phi_k, bigramsPerClass, totalDistinctBigrams):
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
		for j in range(len(words)-1):
			bigram = ' '.join(words[j:j+2])
			for label in classLabels:
				prediction[label] += math.log(probabilityBigramGivenClass(bigram, words[j], label, countOfBigrams, countOfWords, totalDistinctBigrams))
		maxValue = prediction['1']
		maxLabel = '1'
		for label in classLabels:
			if prediction[label] > maxValue:
				maxLabel = label
				maxValue = prediction[label]
		if maxLabel == testLabel[i].strip():
			correct += 1
		#if int(maxLabel) < 5 and int(testLabel[i].strip()) < 5:
		#	correct += 1
		#if int(maxLabel) > 5 and int(testLabel[i].strip()) > 5:
		#	correct += 1
		confusionMatrix[classLabels.index(maxLabel)][classLabels.index(testLabel[i].strip())] += 1
		total += 1
	print('Number of correct predictions : ', correct)
	print('Total number of tests : ', total)
	print('Accuracy = {}'.format(correct*100/total))
	print('Confusion Matrix- ')
	print(confusionMatrix)
	print(sum(confusionMatrix[[0,1,2,3,4,5,6,7],[0,1,2,3,4,5,6,7]]))


classLabels = ['1','2','3','4','7','8','9','10']

# -------------------------------------------------------------------------------------------------------------- #

countOfBigrams = {}
countOfWords = {}
numberOfRecords = {}
for label in classLabels:
	countOfBigrams[label] = {}
	countOfWords[label] = {}
reviewCount, bigramCount = readFiles('./imdbDataset/training.txt', './imdbDataset/imdb_train_labels.txt', countOfBigrams, countOfWords, numberOfRecords)
phi_k = classProbabilities(numberOfRecords, reviewCount)
bigramsPerClass = getBigramsPerClass(countOfBigrams)
totalDistinctWords = getNumberOfWordsInVocab(countOfWords)
test('./imdbDataset/test.txt','./imdbDataset/imdb_test_labels.txt', classLabels, countOfBigrams, countOfWords, numberOfRecords, reviewCount, phi_k, bigramsPerClass, totalDistinctWords)
test('./imdbDataset/training.txt','./imdbDataset/imdb_train_labels.txt', classLabels, countOfBigrams, countOfWords, numberOfRecords, reviewCount, phi_k, bigramsPerClass, totalDistinctWords)
