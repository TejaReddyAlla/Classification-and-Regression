import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys

def ldaLearn(X, y):
    
    #print(np.matrix(X))
    #print(np.matrix(y))
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix
    
    classes = np.unique(y)
    
    means = np.zeros(shape = (X.shape[1], len(classes)))
    
    for c in classes:
    
        c = int(c)
    
        means[:,c-1] = np.mean(X[y[:,0] == c], axis=0)
    
    covmat = np.cov(X.transpose())
    
    return means, covmat


def qdaLearn(X, y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes
    
    # IMPLEMENT THIS METHOD
    
    classes = np.unique(y)
    
    means = np.ones(shape = (X.shape[1], len(classes)))
    
    covmats = []

    for c in classes:
        
        means[:,int(c)-1] = np.mean(X[y[:,0] == c], axis=0)
        
        covmats.append(np.cov(X[y[:,0] == c].transpose()))

    return means, covmats


def ldaTest(means, covmat, Xtest, ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels
    
    # IMPLEMENT THIS METHOD
    
    result = 0

    classes = np.shape(means)[1]

    covInv = np.linalg.inv(covmat)

    tPDF = np.zeros(shape=(Xtest.shape[0], classes))
    
    for classIndex in range(0, classes):

        for index in range(0, Xtest.shape[0]):

            XtSMean = Xtest[index,:]-(means[:, classIndex]).transpose()

            PDF = np.dot(np.dot(XtSMean, covInv), (XtSMean))

            tPDF[index, classIndex] = PDF

    predClasses = np.zeros(shape=(Xtest.shape[0], 1))

    predClasses = (np.argmin(tPDF, axis=1)) + 1

    for index in range(0, Xtest.shape[0]):

        if(ytest[index] == predClasses[index]):

            result = result + 1

        ypred = predClasses.reshape(Xtest.shape[0], 1)
    
    acc = (result/len(ytest))*100

    return acc, ypred


def qdaTest(means, covmats, Xtest, ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels
    
    # IMPLEMENT THIS METHOD

    index = 0

    ypred = np.zeros(shape=(Xtest.shape[0]))
    
    tPDF = np.zeros(means.shape[1])
        
    for XtE in Xtest:
    
        for i in range (0, means.shape[1]):
    
            xSubMean = XtE - means.transpose()[i]
    
            covInv = np.linalg.inv(covmats[i])
    
            tPDF[i] = np.exp(-0.5 * np.dot(np.dot(xSubMean, covInv).transpose(), xSubMean)) / np.sqrt(np.linalg.det(covmats[i]))
    
            mPDF = np.argmax(tPDF)
    
        ypred[index] = mPDF + 1
    
        index = index + 1
    
    matches = 0

    for index in range(0, len(Xtest)):
    
        if ypred[index] == ytest[index][0]:
    
            matches = matches + 1
    
    acc = (matches/len(ytest)) * 100
    
    return acc, ypred


def learnOLERegression(X, y):
    # Inputs:
    # X = N x d
    # y = N x 1
    # Output:
    # w = d x 1
    
    # IMPLEMENT THIS METHOD

    return np.dot(np.linalg.pinv(X), y)


def learnRidgeRegression(X, y, lambd):
    # Inputs:
    # X = N x d
    # y = N x 1
    # lambd = ridge parameter (scalar)
    # Output:
    # w = d x 1
    
    # IMPLEMENT THIS METHOD

    I = np.identity(X.shape[1])

    XT = np.transpose(X)

    return np.dot(np.linalg.inv((np.dot(XT, X) + I*lambd)), np.dot(XT, y))


def testOLERegression(w, Xtest, ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # mse
    
    # IMPLEMENT THIS METHOD
    
    return np.sum(np.square(np.dot(Xtest,w) - ytest))/Xtest.shape[0]


def regressionObjVal(w, X, y, lambd):
    
    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda
    # IMPLEMENT THIS METHOD

    w = np.reshape(w, [len(X.T), 1])

    ySXdw = y - np.dot(X, w)

    ySXdw = np.dot(ySXdw.T, ySXdw) / 2

    lWTdwD2 = (lambd * np.dot(w.T, w)) / 2 

    err = np.add(ySXdw, lWTdwD2)

    XwSy = np.dot(X, w) - y

    XwSy = np.dot(X.T, XwSy)

    errG = (XwSy + (lambd * w)).flatten()

    return err, errG


def mapNonLinear(x, p):
    # Inputs:
    # x - a single column vector (N x 1)
    # p - integer (>= 0)
    # Outputs:
    # Xp - (N x (p+1))
    
    # IMPLEMENT THIS METHOD

    xp = np.ones((x.shape[0], p+1))

    for pi in range(1, p+1):

        xp[:, pi] = x**pi

    return xp

# Main script

# Problem 1
# load the sample data
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'),encoding = 'latin1')

# LDA
means,covmat = ldaLearn(X,y)
ldaacc,ldares = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))
# QDA
means,covmats = qdaLearn(X,y)
qdaacc,qdares = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))

# plotting boundaries
x1 = np.linspace(-5,20,100)
x2 = np.linspace(-5,20,100)
xx1,xx2 = np.meshgrid(x1,x2)
xx = np.zeros((x1.shape[0]*x2.shape[0],2))
xx[:,0] = xx1.ravel()
xx[:,1] = xx2.ravel()

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)

zacc,zldares = ldaTest(means,covmat,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zldares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest.flatten())
plt.title('LDA')

plt.subplot(1, 2, 2)

zacc,zqdares = qdaTest(means,covmats,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zqdares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest.ravel())
plt.title('QDA')

plt.show()
# Problem 2
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'),encoding = 'latin1')

# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
#Xtest_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)     <=   For training data
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)  # <=  For testing data

w = learnOLERegression(X,y)
mle = testOLERegression(w,Xtest,ytest)

w_i = learnOLERegression(X_i,y)
#mle_i = testOLERegression(w_i,Xtest_i,y)   <=   For training data
mle_i = testOLERegression(w_i,Xtest_i,ytest)  # <=  For testing data

print('MSE without intercept '+str(mle))
print('MSE with intercept '+str(mle_i))

# Problem 3
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses3_train = np.zeros((k,1))
mses3 = np.zeros((k,1))
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    mses3_train[i] = testOLERegression(w_l,X_i,y)
    mses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    #print(lambd, "   ", mses3_train[i], "   ", mses3[i]) # <=  For printing lamdba for testing and training
    i = i + 1
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.subplot(1, 2, 2)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')

plt.show()

# For plotting comparisions
""" 
plt.plot(w_i, color='r')
plt.plot(w_i, 'r*')
plt.plot(w_l, color='b')
plt.plot(w_l, 'b*')
plt.show()

for s_no in range(0, len(w_l)):
    print(s_no, " ", w_i[s_no][0], " ", w_l[s_no][0])
 """

# Problem 4
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses4_train = np.zeros((k,1))
mses4 = np.zeros((k,1))
opts = {'maxiter' : 20}    # Preferred value.
w_init = np.ones((X_i.shape[1],1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
    w_l = np.transpose(np.array(w_l.x))
    w_l = np.reshape(w_l,[len(w_l),1])
    mses4_train[i] = testOLERegression(w_l,X_i,y)
    mses4[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses4_train)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.legend(['Using scipy.minimize','Direct minimization'])

plt.subplot(1, 2, 2)
plt.plot(lambdas,mses4)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')
plt.legend(['Using scipy.minimize','Direct minimization'])
plt.show()


# Problem 5
pmax = 7
lambda_opt = 0.06 # REPLACE THIS WITH lambda_opt estimated from Problem 3
mses5_train = np.zeros((pmax,2))
mses5 = np.zeros((pmax,2))
for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    mses5_train[p,0] = testOLERegression(w_d1,Xd,y)
    mses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    mses5_train[p,1] = testOLERegression(w_d2,Xd,y)
    mses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)
    #print(p, " ", mses5[p,0], " ", mses5_train[p,0], " ", mses5[p,1], " ", mses5_train[p,1])   #  <= To check for best p

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(range(pmax), mses5_train)
#plt.plot(mses5_train, 'g*')   #  <= Plot points with star
#plt.plot(mses5_train[:,1], 'r*')   #  <= Plot points with star
plt.title('MSE for Train Data')
plt.legend(('No Regularization','Regularization'))
plt.subplot(1, 2, 2)
plt.plot(range(pmax), mses5)
#plt.plot(mses5, 'g*')   #  <= Plot points with star
#plt.plot(mses5[:,1], 'r*')   #  <= Plot points with star
plt.title('MSE for Test Data')
plt.legend(('No Regularization','Regularization'))
plt.show()

