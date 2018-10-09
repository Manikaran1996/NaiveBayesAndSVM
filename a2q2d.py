from svmutil import *
from random import shuffle
import matplotlib.pyplot as plt
import math


k = 10
g = 0.05
C = [0.00001, 0.001, 1, 5, 10]

# data
trainFileScaled = '/home/manikaran/Assignment2/mnist/trainLibSVMScaled.txt'
testFileScaled = '/home/manikaran/Assignment2/mnist/testLibSVMScaled.txt'
yTrain, xTrain = svm_read_problem(trainFileScaled)

result = []
problem = svm_problem(yTrain,xTrain)
paramString = '-t 2 -c {} -g {} -v 10'.format(C[0], g)
parameter = svm_parameter(paramString)
#print(len(y), len(x))
result.append(svm_train(problem,parameter))

paramString = '-t 2 -c {} -g {} -v 10'.format(C[1], g)
parameter = svm_parameter(paramString)
#print(len(y), len(x))
result.append(svm_train(problem,parameter))

paramString = '-t 2 -c {} -g {} -v 10'.format(C[2], g)
parameter = svm_parameter(paramString)
#print(len(y), len(x))
result.append(svm_train(problem,parameter))

paramString = '-t 2 -c {} -g {} -v 10'.format(C[3], g)
parameter = svm_parameter(paramString)
#print(len(y), len(x))
result.append(svm_train(problem,parameter))

paramString = '-t 2 -c {} -g {} -v 10'.format(C[4], g)
parameter = svm_parameter(paramString)
#print(len(y), len(x))
result.append(svm_train(problem,parameter))

with open('ouput.txt', 'w') as fout:
    for x in result:
        print(x, file=fout)

models = []
acc = []
yTest, xTest = svm_read_problem(testFileScaled)
problem = svm_problem(yTrain,xTrain)
for c in C:
    paramString = '-t 2 -c {} -g {}'.format(c, g)
    parameter = svm_parameter(paramString)
    model = svm_train(problem,parameter)
    models.append(model)
    z = svm_predict(yTest, xTest, model)
    acc.append(z)
    
import os
path = '/home/manikaran/Assignment2/q2d2/'
for i in range(5):
    os.makedirs(path + 'C_{}'.format(C[i]))

for i in range(len(C)):
    svm_save_model(path + 'C_{}/model'.format(C[i]), models[i])

C = [0.00001, 0.001, 1, 5, 10]
validationSetAccuracy = [71.555, 71.54, 97.4, 97.535, 97.46] 
testSetAccuracy = [72.1, 72.1, 97.23, 97.29, 97.29]
C = list(map(lambda x: math.log(x), C))
plt.figure()
plt.plot(C, validationSetAccuracy)
plt.plot(C, testSetAccuracy)
plt.show()