import numpy as np
import pandas as pd
from sklearn.utils import shuffle

def init_weight_and_bias(M1, M2):
    W = np.random.rand(M1, M2) / np.sqrt(M1)
    b = np.zeros(M2)
    return W.astype(np.float32), b.astype(np.float32)

def init_filter(shape, poolsz):
    # used in convolutional neural networks
    w = np.random.randn(*shape) / np.sqrt(
        np.prod(shape[1:]) + shape[0]*np.prod(shape[2:] / np.prod(poolsz)))
    return w.astype(np.float32)

def relu(x):
    return  x * (x>0)

def softmax(A):
    expA = np.exp(A)
    return expA / expA.sum(axis=1, keepdims=True)

def sigmoid_cost(T, Y):
    return -(T*np.log(Y) + (1-T)*np.log(1-Y)).sum()

def cost(T, Y):
    return -(T*np.log(Y)).sum()

def cost2(T, Y):
    # softmax crossentropy using actual values, but should yield same answer as cost
    N = len(T)
    return -np.log(Y[np.arange(N), T]).sum()

def error_rate(targets, predictions):
    return np.mean(targets != predictions)

def y2indicator(y):
    N = len(y)
    K = len(set(y))
    ind = np.zeros((N, K))
    for i in range(N):
        ind[i, y[i]] = 1
    return ind

def getData(balance_ones=True, Ntest=1000):
    # images are 48x48 = 2304 size vectors
    Y = []
    X = []
    first = True
    csvdata = open('fer2013.csv')
    for line in csvdata:
        if first:
            first = False
        else:
            row = line.split(',')
            Y.append(int(row[0]))
            X.append([int(p) for p in row[1].split()])

    X, Y = np.asarray(X) / 255.0, np.array(Y)

    # shuffle and split
    X, Y = shuffle(X, Y) # type: ignore
    Xtrain, Ytrain = X[:-Ntest], Y[:-Ntest] # type: ignore
    Xvalid, Yvalid = X[-Ntest:], Y[-Ntest:] # type: ignore

    if balance_ones:
        # balance the 1 class
        X0, Y0 = Xtrain[Ytrain!=1, :], Ytrain[Ytrain!=1]
        X1 = Xtrain[Ytrain==1, :]
        X1 = np.repeat(X1, 9, axis=0)
        Xtrain = np.vstack([X0, X1])
        Ytrain = np.concatenate((Y0, [1]*len(X1)))

    return Xtrain, Ytrain, Xvalid, Yvalid

def getImageData():
    X, Y, _, _ = getData()
    N, D = X.shape
    d = int(np.sqrt(D))
    X = X.reshape(N, 1, d, d)
    return X, Y

def getBinaryData():
    Y = []
    X = []
    first = True
    csvdata = open('fer2013.csv')
    for line in csvdata:
        if first:
            first = False
        else:
            row = line.split(',')
            y = int(row[0])
            if y == 0 or y == 1:
                Y.append(y)
                X.append([int(p) for p in row[1].split()])
    return np.asarray(X)/255., np.array(Y)


