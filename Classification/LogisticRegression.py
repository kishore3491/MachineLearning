import numpy as np

"""
ToDos:
1. Regularization
2. Polynomial features
3. Stochastic Gradient Descent or others.
4. Debugging Gradient Descent without Normalization.
"""

class LogisticRegression:
    """
    A simplest classification algorithm, which uses:
    1. Sigmoid function to transform hypothesis
    2. Batch Gradient Descent (Upgrade to Stochastic | Make it agnostic)
    Note:
    NOT adjusted for bias.
    DOES NOT normalizes given dataset.
    """
    def __init__(self):
        self.weights = None
        self.__m__ = None
        self.__n__ = None
        self.__alpha__ = None
        self.__gradientStoppingDiff__ = None
        self.costVector = None

    def sigmoid(self, X):
        return float(1/(1 + np.exp(-X)))

    def hypothesis(self, W, X) :
        return self.sigmoid(np.dot(W, X))

    def costFunction(self, W, X, Y):
        res = 0
        error = 0.0001
        for i in range(self.__m__):
            hi = self.hypothesis(W, X[i, :])
            if hi == 1:
                hi -= error
            elif hi == 0:
                hi += error
            res -= ((Y[i]*np.log(hi)) + ((1 - Y[i])*(np.log(1 - hi))))

        return float(res/self.__m__)

    def gradientDescent(self, W, X, Y, alpha, stopping_diff, debug):
        diff = 1
        prev_cost = self.costFunction(W, X, Y)
        new_W = np.zeros(self.__n__)

        if debug:
            costVector = []

        epoch = 0
        while diff > stopping_diff:
            for j in range(self.__n__):
                dPen = 0

                for i in range(self.__m__):
                    dPen += (self.hypothesis(W, X[i, :]) - Y[i])*X[i][j]
                dPen /= self.__m__

                new_W[j] = W[j] - alpha*(dPen)

            W = new_W
            new_cost = self.costFunction(W, X, Y)
            diff = abs(new_cost - prev_cost)
            prev_cost = new_cost
            epoch += 1
            print ("Epoch #{0}, loss: {1}".format(epoch, new_cost))

            if debug:
                costVector.append(new_cost)
        if debug:
            return (W, costVector)
        else:
            return W

    def fit(self, X, Y,  alpha=0.001, stopping_diff=0.001, debug=False):
        (m,n) = X.shape
        self.__m__ = m
        self.__n__ = n
        self.__alpha__ = alpha
        self.__gradientStoppingDiff__ = stopping_diff
        W = np.zeros(n)

        if debug:
            (self.weights, self.costVector) = self.gradientDescent(W, X, Y, alpha=alpha, stopping_diff=stopping_diff, debug=debug)
        else:
            self.weights = self.gradientDescent(W, X, Y, alpha=alpha, stopping_diff=stopping_diff, debug=False)

    def test(self, Xtest):
        if self.weights is not None:
            (m, n) = Xtest.shape
            assert n == self.__n__
            res = []
            for i in range(m):
                y = self.hypothesis(self.weights, Xtest[i, :])
                if y > 0.5:
                    res.append(1)
                else:
                    res.append(0)

            return res
        else:
            raise ValueError("Please fit the model first.")
