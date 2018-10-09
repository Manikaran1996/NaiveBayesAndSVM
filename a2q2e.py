from svmutil import *
import numpy
import matplotlib.pyplot as plt

def generateConfusionMatrix(predictedLabels, originalLabels):
    confusionMatrix = [[0]*10 for i in range(10)]
    for predicted, original in zip(predictedLabels, originalLabels):
        confusionMatrix[int(predicted)][int(original)] += 1
    return confusionMatrix

trainFileScaled = '/home/manikaran/M.tech/MachineLearning/Assignment2/mnist/trainLibSVMScaled.txt'
testFileScaled = '/home/manikaran/M.tech/MachineLearning/Assignment2/mnist/testLibSVMScaled.txt'

yTrain, xTrain = svm_read_problem(trainFileScaled)

problem = svm_problem(yTrain, xTrain)
paramString = '-t 2 -g 0.05 -c 5'
params = svm_parameter(paramString)
model = svm_train(problem, params)

yTest, xTest = svm_read_problem(testFileScaled)
(label, acc, vals) = svm_predict(yTest, xTest, model)

svm_save_model('trainedModelq2e', model)

with open('/home/manikaran/M.tech/MachineLearning/Assignment2/mnist/test.csv','r') as testImageVectors:
    imageVectors = numpy.loadtxt(testImageVectors, delimiter=',' , dtype=int)
    wrongPredictions = []
    featureXOfWrongPredictions = []
    count = 0
    labelIndex = imageVectors.shape[1]-1
    for i in range(len(label)):
        if yTest[i] != label[i]:
            count += 1
            wrongPredictions.append((yTest[i], label[i]))
            featureXOfWrongPredictions.append(imageVectors[i, :labelIndex])
    print("Number of wrong predictions = ", count)

plotMe = []
for i,(actual, test) in enumerate(wrongPredictions):
    if test == 8 and actual == 2:
        plotMe.append((test,actual,featureXOfWrongPredictions[i]))
    elif test == 2 and actual == 7:
        plotMe.append((test,actual,featureXOfWrongPredictions[i]))

plt.title('Predicted : ' + str(plotMe[0][0]) + " Actual : " + str(plotMe[0][1]))
plt.imshow(plotMe[0][2].reshape(28,28))

plt.title('Predicted : ' + str(plotMe[1][0]) + " Actual : " + str(plotMe[1][1]))
plt.imshow(plotMe[1][2].reshape(28,28))

plt.title('Predicted : ' + str(plotMe[2][0]) + " Actual : " + str(plotMe[2][1]))
plt.imshow(plotMe[2][2].reshape(28,28))

confusionMatrix = generateConfusionMatrix(label, yTest)
for row in confusionMatrix:
    for x in row:
        print('{:4}'.format(x), end = ' ')
    print()
