import unittest
import numpy as np

class TestFeedForward(unittest.TestCase):

    def setUp(self):
        data = self.loadTestData()
        self.Y = data[:, 2]
        X = data[:, [0, 1]]
        self.X = self.normalize(X)
        import imp
        foo = imp.load_source('FeedForwardBackProp', '/home/kbanala/Programming/MachineLearning/NeuralNets/FeedForward.py')
        self.model = foo.FeedForwardBackProp(size=[2, 3, 1])
        # self.model.fit_SGD(self.X, self.Y, alpha=4, batch=10, debug=True)

    # TODO Move to common utils
    def normalize(self, X):
        mean = np.mean(X, axis=0)
        sigma = np.std(X, axis=0)

        try:
            (m,n) = X.shape
        except ValueError:
            (m,n) = (len(X), None)
        X_norm = X
        if (n):
            for i in range(m):
                for j in range(n):
                    X_norm[i][j] = float((X[i][j] - mean[j])/sigma[j])
        else:
            X_norm = (X - mean)/sigma

        return X_norm

    def test_forward_and_back_pass(self):
        (A, W, Z, D, gradients) = self.model.init_caches()
        # Let W be a constant to get expected result
        W = []
        sizes = self.model.size
        for i in range(len(sizes) - 1):
            W.append(np.ones([sizes[i+1], sizes[i]]))
        A[0] = self.X[1]
        self.model.forward_pass(A, D, W, Z)
        exp_A = [np.array([-1.82625564, -1.2075414 ]),
                np.array([ 0.04592218,  0.04592218,  0.04592218]),
                np.array([ 0.53438726])]
        # Complex operation, which requires each unit in Neural Net
        # to be logically close to expected values.
        for i in range(len(sizes)):
            res = np.isclose(A[i], exp_A[i])
            for val in res:
                self.assertTrue(val)

        # Do final layer's error
        L = len(A) - 1
        D[L-1] = A[L] - self.Y[1]
        # gradients for last layer
        gradients[L-1] += np.outer(A[L-1], D[L-1]).T
        self.model.backprop(A, D, W, Z, gradients)

        exp_D = [np.array([ 0.02341329,  0.02341329,  0.02341329]), np.array([ 0.53438726])]
        for i in range(len(D)):
            res = np.isclose(D[i], exp_D[i])
            for val in res:
                self.assertTrue(val)
        exp_gradients = [np.array([[-0.04275865, -0.02827251],
                [-0.04275865, -0.02827251],
                [-0.04275865, -0.02827251]]
            ),
            np.array([[ 0.02454023,  0.02454023,  0.02454023]])
        ]
        for i in range(len(gradients)):
            res = np.isclose(gradients[i], exp_gradients[i])
            for val in res:
                self.assertTrue(val.all)

    def test_batchGradientDescent(self):
        pass

    def test_SGD(self):
        pass

    def loadTestData(self):
        # test data set
        fname = '/media/kbanala/361ED9C21ED97AF7/Programming/MachineLearning/Coursera/machine-learning-ex2/ex2/ex2data1.txt'
        return np.loadtxt(fname, delimiter=',')

if __name__ == '__main__':
    unittest.main()
