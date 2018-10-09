from svmutil import *
import sys
import numpy
import pickle

def convert(inputData, outputData):
	lines = ''
	output = open('temp.txt', 'w')
	for i in range(inputData.shape[0]):
		line = str(outputData[i])
		for j in range(inputData.shape[1]):
			if inputData[i,j] == 0:
			    continue
			line += ' {}:{}'.format(j+1, inputData[i,j]/255)
		print(line, file = output)
	output.close()

def test(classifiers, inData, outData):
    predicted = []
    for i in range(inData.shape[0]):
        count = [0]*10
        for j in range(10):
            for k in range(j):
                if j == k:
                    continue
                val = numpy.dot(inData[i, :],classifiers[j][k][0]) + classifiers[j][k][1]
                if val > 0:
                    count[j] += 1
                else:
                    count[k] += 1
        maxIndex = 0
        maxVal = count[0]
        for j in range(10):
            if maxVal <= count[j]:
                maxIndex = j
                maxVal = count[j]
        predicted.append(maxIndex)
    sum_ = 0
    for i in range(outData.shape[0]):
        if outData[i] == predicted[i]:
            sum_ += 1
    return sum_*100/(inData.shape[0]), predicted


inputFile = sys.argv[2]
outputFile = sys.argv[3]
inputData = numpy.loadtxt(inputFile, delimiter=',')
outputData = numpy.loadtxt(outputFile, delimiter=',')

if int(sys.argv[1]) == 1:
	classifierFileName = 'pegasos.pickle'
	classifierFile = open(classifierFileName, 'rb')
	model = pickle.load(classifierFile)
	classifiers = model['classifiers']
	acc, predictions = test(classifiers, inputData, outputData)

elif int(sys.argv[1]) == 2:
	model = svm_load_model('linearModel')
	convert(inputData, outputData)
	y,x = svm_read_problem('temp.txt')
	predictions, acc,_ = svm_predict(y,x,model)
	acc = acc[0]

elif int(sys.argv[1]) == 3:
	model = svm_load_model('trainedModelq2e')
	convert(inputData, outputData)
	y,x = svm_read_problem('temp.txt')
	predictions, acc,_ = svm_predict(y,x,model)
	acc = acc[0]


print('Accuracy : ' , acc)
outFile = open('predicted_text.txt', 'w')
for prediction in predictions:
	print(prediction, file=outFile)
outFile.close()