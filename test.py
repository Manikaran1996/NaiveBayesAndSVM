import pickle
import sys
import numpy
import math

def probabilityWordGivenClass(word, classLabel, count, wordsPerClass, numberOfwords):
	numerator = count[classLabel].get(word,0) + 1
	denominator = wordsPerClass[classLabel] + numberOfwords
	return numerator/denominator

def probabilityBigramGivenClass(bigram, word, classLabel, countBigrams, countWords, numberOfwords):
	numerator = countBigrams[classLabel].get(bigram,0) + 1
	denominator = countWords[classLabel].get(word,0) + numberOfwords
	return numerator/denominator

def test1(testDataFileName, testLabelFileName, classLabels, countOfWords, numberOfRecords, reviewCount, phi_k, wordsPerClass, totalDistinctWords):
	confusionMatrix = numpy.zeros((8,8), dtype=int)
	testDataFile = open(testDataFileName, 'r')
	testLabelFile = open(testLabelFileName, 'r')
	testData = testDataFile.readlines()
	testLabel = testLabelFile.readlines()
	testDataFile.close()
	testLabelFile.close()
	correct = 0
	total = 0
	predictions = []
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
		predictions.append(maxLabel)
	print('Number of correct predictions : ', correct)
	print('Total number of tests : ', total)
	print('Accuracy = {}'.format(correct*100/total))
	print('Confusion Matrix- ')
	print(confusionMatrix)
	#print(sum(confusionMatrix[[0,1,2,3,4,5,6,7],[0,1,2,3,4,5,6,7]]))
	return predictions

def test2(testDataFileName, testLabelFileName, classLabels, countOfWords, 
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
	predictions = []
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
		predictions.append(maxLabel)
		confusionMatrix[classLabels.index(maxLabel)][classLabels.index(testLabel[i].strip())] += 1
		total += 1
	print('Number of correct predictions : ', correct)
	print('Total number of tests : ', total)
	print('Accuracy = {}'.format(correct*100/total))
	print('Confusion Matrix- ')
	print(confusionMatrix)
	return predictions
	#print(sum(confusionMatrix[[0,1,2,3,4,5,6,7],[0,1,2,3,4,5,6,7]])) 

if int(sys.argv[1]) == 1:
	pickle_in = open('model1.pickle', 'rb')

elif int(sys.argv[1]) == 2:
	pickle_in = open('model2.pickle', 'rb')

elif int(sys.argv[1]) == 3:
	pickle_in = open('bestModel.pickle','rb')

classLabels = ['1','2','3','4','7','8','9','10']
params = pickle.load(pickle_in)
reviewCount = params['reviewCount'] 
countOfWords = params['countOfWords']
numberOfRecords = params['numberOfRecords']
phi_k = params['phi_k']
wordsPerClass = params['wordsPerClass'] 
totalDistinctWords = params['totalDistinctWords']

if int(sys.argv[1]) <3:
	predictions = test1(sys.argv[2], sys.argv[3], classLabels, countOfWords, numberOfRecords, reviewCount, phi_k, wordsPerClass, totalDistinctWords)
else :
	countOfBigrams = params['countOfBigrams']
	predictions = test2(sys.argv[2], sys.argv[3], classLabels, countOfWords, countOfBigrams, numberOfRecords, reviewCount, phi_k, wordsPerClass, totalDistinctWords)

outFile = open('predicted_text.txt', 'w')
for prediction in predictions:
	print(prediction, file=outFile)
outFile.close()