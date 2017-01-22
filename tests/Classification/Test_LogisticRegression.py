import unittest
import numpy as np

class TestLogisticRegression(unittest.TestCase):

    def setUp(self):
        data = self.loadTestData()
        self.Y = data[:, 2]
        X = data[:, [0, 1]]
        X = self.normalize(X)
        self.X = np.hstack((np.ones((len(X), 1)), X))
        import imp
        foo = imp.load_source('LogisticRegression', '/home/kbanala/Programming/MachineLearning/Classification/LogisticRegression.py')
        self.model = foo.LogisticRegression()
        self.model.fit_SGD(self.X, self.Y, alpha=4, batch=10, debug=True)

    def test_modelCost(self):
        """
        Make sure final training cost is less than 0.25
        Ideally, it should be less than 0.21 for this dataset.
        """
        self.assertTrue(self.model.costVector[-1] < 0.25)

    def test_SampleData(self):
        """
        For given dataset, make sure atleast 9 classifications
        are correct from last ten.
        TODO Testing against trained data. Get more test data or change dataset.
        """
        Xtest = self.X[-10:, :]
        res = self.model.test(Xtest)
        self.assertTrue(np.sum((res - self.Y[-10:])**2) < 2)

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

    def loadTestData(self):
        # test data set
        fname = '/media/kbanala/361ED9C21ED97AF7/Programming/MachineLearning/Coursera/machine-learning-ex2/ex2/ex2data1.txt'
        return np.loadtxt(fname, delimiter=',')

if __name__ == '__main__':
    unittest.main()
