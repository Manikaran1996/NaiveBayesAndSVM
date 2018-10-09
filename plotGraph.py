C = [0.00001, 0.001, 1, 5, 10]
import matplotlib.pyplot as plt
import math
validationSetAccuracy = [71.555, 71.54, 97.4, 97.535, 97.46] 
testSetAccuracy = [72.1, 72.1, 97.23, 97.29, 97.29]
C = list(map(lambda x: math.log(x), C))
plt.figure()
ax = plt.subplot('111')
plt.plot(C, validationSetAccuracy, label='Validation Set')
plt.plot(C, testSetAccuracy, label='Test Set')
plt.title('C vs Accuracy')
ax.set_xlabel('C')
ax.set_ylabel('Accuracy')
plt.legend()
plt.show()