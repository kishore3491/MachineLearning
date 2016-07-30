from math import exp, log
import numpy as np
import matplotlib.pyplot as plt

# TODO
# There are errors.
# Can't fix it until visualization is available.
# Change to a class and do testing outside this module.

def sigmoid(x):
    return float(1/(1 + exp(-x)))

def predict(theta, X):
    m = len(X)
    p = np.zeros(m)
    for i in range(m):
        if sigmoid(theta.dot(X[i])) >= 0.5:
            p[i] = 1
    return p

def hypothesis(theta, X):
    return sigmoid(theta.dot(X))

def costFunction(theta, X, Y):
    m = len(X)
    res = 0
    for i in range(m):
        hypVal = sigmoid(theta.dot(X[i]))
        res += (Y[i]*log(hypVal) + (1 - Y[i])*log(1 - hypVal))
    return float(-res/m)


def logistic_gradientDescent(theta, X, Y, alpha=0.01, num_iters=1000):
    (m, n) = X.shape

    hx = np.zeros((m, 1))

    for i in range(num_iters):
        for i in range(m):
            hx[i] = sigmoid(theta.dot(X[i]))
        for j in range(n):
            gradient = 0
            for i in range(m):
                # TODO hypothesis is computed n times here. Try to do it only once.
                gradient += (hx[i] - Y[i])*X[i][j]
            theta[j] -= alpha*(gradient/m)

    return theta

def plotData(X, Y):
    pass

if __name__ == "__main__":
    fname = 'D:\Programming\MachineLearning\Coursera\machine-learning-ex2\ex2\ex2data1.txt'
    data = np.loadtxt(fname, delimiter=',')
    X = data[:, [0,1]]
    Y = data[:, [2]]

    X = np.append(np.ones((len(X), 1)), X, axis=1)

    print ('X, Y: ', X[5], Y[5])

    # TODO plot against cost function and initial pos, neg values.

    (m,n) = X.shape

    init_theta = np.zeros(n)

    print ('Initial cost: ', costFunction(init_theta, X, Y))

    theta = logistic_gradientDescent(init_theta, X, Y, num_iters=100)

    print ('theta: ', theta)

    print ('Final cost on training data: ', costFunction(theta, X, Y))

    acc = predict(theta, X)
    print ('Accuracy on training set: ', acc)

    X_test = np.array([[1.0, 10.0, 16.0]])
    res = predict(theta, X_test)
    print ('Predicted value: ', res)
