import numpy
import math

def readFile(fileName):
	data = numpy.loadtxt(fileName, delimiter=',', dtype=int)
	return data

def pegasosMiniBatch(data, noOfIter=1000, batchSize=100, lambda_=1):
    m = numpy.arange(data.shape[0])
    w = numpy.zeros((data.shape[1]-1,1))
    b = 0
    labelIndex = data.shape[1]-1
    for i in range(1,noOfIter+1):
        numpy.random.shuffle(m)
        a_t = m[:batchSize]
        examplesForProcessing = data[a_t]
        #print("Batch shape : ", examplesForProcessing.shape) 
        # y*(x'.w + b)
        result = examplesForProcessing[:,labelIndex:]*(numpy.dot(examplesForProcessing[:,:labelIndex], w) + b) 
        #print("Result shape : ", result.shape)
        # selecting examples for which y*(x'.w + b) < 1
        filtered = examplesForProcessing[result.flatten()<1]
        #print("Filtered shape : ", filtered.shape)
        # xi * yi
        result = filtered[:,labelIndex:]*filtered[:,:labelIndex]
        # sumOf xi * yi
        result = result.sum(axis=0, keepdims=True)
        #print("Result shape : ", result.shape)
        # learning-rate
        eta = 1/(lambda_*i)
        # updating params
        w1 = (1-eta*lambda_)*w + (eta/batchSize)*result.T
        if math.sqrt(numpy.dot((w1 - w).T, (w1-w))) < 0.000001:
            print('Stopped in ', i, 'th iteration.')
            break
        w = w1
        b = b + (eta/batchSize)*filtered[:,labelIndex].sum()
        #print("W shape : " ,w.shape)
    return (w,b)

def trainOneVsOneClassifiers(trainingData, noOfIter = 1000):
    labelIndex = trainingData.shape[1]-1
    classifiers = [[None]*10 for i in range(10)]
    labels = trainingData[:,labelIndex]
    for i in range(10):
        data1 = trainingData[labels == i]
        data1[:,labelIndex] = 1
        for j in range(i):
            if i == j:
                continue
            data2 = trainingData[labels == j]
            data2[:,labelIndex] = -1
            data = numpy.concatenate((data1, data2))
            classifiers[i][j] = pegasosMiniBatch(data, noOfIter)
    return classifiers

def test(classifiers, testingData):
    labelIndex = testingData.shape[1] - 1
    predicted = []
    for i in range(testingData.shape[0]):
        count = [0]*10
        for j in range(10):
            for k in range(j):
                if j == k:
                    continue
                val = numpy.dot(testingData[i, :labelIndex],classifiers[j][k][0]) + classifiers[j][k][1]
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
    for i in range(testingData.shape[0]):
        if testingData[i,labelIndex] == predicted[i]:
            sum_ += 1
    return sum_*100/(testingData.shape[0])


trainingData = readFile('./mnist/train.csv')
testingData = readFile('./mnist/test.csv')
classifiers = trainOneVsOneClassifiers(trainingData, noOfIter=1000)
print(test(classifiers, testingData))

