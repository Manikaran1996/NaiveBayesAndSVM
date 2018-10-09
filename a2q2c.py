from svmutil import *

trainFile = '/home/manikaran/M.tech/MachineLearning/Assignment2/mnist/trainLibSVMScaled.txt'
testFile = '/home/manikaran/M.tech/MachineLearning/Assignment2/mnist/testLibSVMScaled.txt'

y,x = svm_read_problem(trainFile)

''' struct svm_problem describes the problem:

    struct svm_problem
    {
        int l;
        double *y;
        struct svm_node **x;
    };
    
    where `l' is the number of training data, and `y' is an array containing
    their target values. (integers in classification, real numbers in
    regression) `x' is an array of pointers, each of which points to a sparse
    representation (array of svm_node) of one training vector.'''

problem = svm_problem(y,x)

''' struct svm_parameter describes the parameters of an SVM model 
options:
-s svm_type : set type of SVM (default 0)
    0 -- C-SVC
    1 -- nu-SVC''
    2 -- one-class SVM
    3 -- epsilon-SVR
    4 -- nu-SVR
-t kernel_type : set type of kernel function (default 2)
    0 -- linear: u'*v
    1 -- polynomial: (gamma*u'*v + coef0)^degree
    2 -- radial basis function: exp(-gamma*|u-v|^2)
    3 -- sigmoid: tanh(gamma*u'*v + coef0)
-d degree : set degree in kernel function (default 3)
-g gamma : set gamma in kernel function (default 1/num_features)
-r coef0 : set coef0 in kernel function (default 0)
-c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)
-n nu : set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)
-p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)
-m cachesize : set cache memory size in MB (default 100)
-e epsilon : set tolerance of termination criterion (default 0.001)
-h shrinking: whether to use the shrinking heuristics, 0 or 1 (default 1)
-b probability_estimates: whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0)
-wi weight: set the parameter C of class i to weight*C, for C-SVC (default 1)
'''
param = svm_parameter('-t 0 -c 1 -g 0.05')

'''Function: struct svm_model *svm_train(const struct svm_problem *prob,
                                const struct svm_parameter *param);
            
    This function constructs and returns an SVM model according to
    the given training data and parameters.
    
    struct svm_model stores the model obtained from the training procedure.
'''

model = svm_train(problem,param)

''' Function: double svm_predict(const struct svm_model *model,
                               const struct svm_node *x);

    This function does classification or regression on a test vector x
    given a model. '''
svm_save_model('linearModel', model)
yTest,xTest = svm_read_problem(testFile)
predicted = svm_predict(yTest, xTest, model)

yScaled,xScaled = svm_read_problem(trainFile)
problemScaled = svm_problem(yScaled,xScaled)
paramScaled = svm_parameter('-t 2 -c 1 -g 0.05')
modelGaussianScaled = svm_train(problemScaled, paramScaled)
yTestScaled,xTestScaled = svm_read_problem(testFile)
predictedScaled = svm_predict(yTestScaled, xTestScaled, modelGaussianScaled)