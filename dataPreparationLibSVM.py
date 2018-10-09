import sys
import numpy

def convert(inputFileName, outputFileName):
    data = numpy.loadtxt(inputFileName, delimiter=',', dtype=float)
    outFile = open(outputFileName, 'w')
    labelIndex = data.shape[1]-1
    for i in range(data.shape[0]):
        line = str(data[i,labelIndex])
        for j in range(data.shape[1]-1):
            if data[i,j] == 0:
                continue
            line += ' {}:{}'.format(j+1, data[i,j]/255)
        print(line, file=outFile)
    outFile.close()
    return data

convert('./mnist/train.csv', './mnist/trainLibSVMScaled.txt')
convert('./mnist/test.csv', './mnist/testLibSVMScaled.txt')
