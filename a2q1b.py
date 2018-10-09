import random
import numpy

testReviewFile = '/home/manikaran/M.tech/MachineLearning/Assignment2/imdbDataset/imdb_test_text.txt'
testLabelFile = '/home/manikaran/M.tech/MachineLearning/Assignment2/imdbDataset/imdb_test_labels.txt'
trainLabelFile = '/home/manikaran/M.tech/MachineLearning/Assignment2/imdbDataset/imdb_train_labels.txt'
classLabels = [1,2,3,4,7,8,9,10]
reviewLabels = (open(testLabelFile, 'r')).readlines()
trainLabels = (open(trainLabelFile, 'r')).readlines()
count = [0]*11
for label in trainLabels:
	count[int(label)] += 1
maxLabel = 0
maxNum = count[0]
for i in range(len(count)):
	if count[i] > maxNum:
		maxLabel = i
		maxNum = count[i]
correctRandomPredictions = 0
correctMajorityPredictions = 0
numberOfExamples = 0
for label in reviewLabels:
	index = random.randint(0,len(classLabels)-1)
	if classLabels[index] == int(label):
		correctRandomPredictions += 1
	if maxLabel == int(label):
		correctMajorityPredictions += 1
	numberOfExamples += 1


print('Number of correct predictions if predicted randomly = ', correctRandomPredictions)
print('Accuracy = ' , (correctRandomPredictions*100)/numberOfExamples)
print('Number of correct predictions if most occurred class is predicted = ', correctMajorityPredictions)
print('Accuracy = ', (correctMajorityPredictions*100)/numberOfExamples)
