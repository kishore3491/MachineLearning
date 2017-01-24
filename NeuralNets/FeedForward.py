import numpy as np

"""
ToDos:
1. Reformat module and class names + code organization.
2. Use higher order functions.
3. Use any cost function.
4. Use any gradient descent variant.
5. Use any activation function.
"""

class FeedForwardBackProp:
    """
    Construct a Feed Forward Neural Net which uses SGD with
    Back Propagation to compute Gradients.
    Note:
    DOES NOT normalizes given dataset.
    """
    def __init__(self, size=[]):
        self.size = size
        self.weights = None
        self.biases = None
        self.__m__ = None
        self.__n__ = None
        self.__alpha__ = None
        self.__gradientStoppingDiff__ = None
        self.costVector = None
        # TODO Create blank network with weights, biases, activations etc.

    def init_caches(self):
        """
        Initialize set of caches to use in-between operations.
        Note: Do not copy from Z or other array by reference
        """
        sizes = self.size

        A = []
        for i in sizes:
            A.append(np.zeros(i))

        Z = []
        for i in range(1, len(sizes)):
            Z.append(np.zeros(sizes[i]))

        # Initialize Error cache
        D = []
        for i in range(1, len(sizes)):
            D.append(np.zeros(sizes[i]))

        # Initialize random weights array and append weights
        W = []
        for i in range(len(sizes) - 1):
            W.append(np.random.rand(sizes[i+1], sizes[i]))

        gradients = []
        for i in range(len(sizes) - 1):
            gradients.append(np.zeros([sizes[i+1], sizes[i]]))

        return (A, W, Z, D, gradients)

    def forward_pass(self, A, D, W, Z):
        """
        A complete pass through network, while computing
        activations for each unit.
        TODO activation function agnostic.
        """
        for i in range(1, len(A)):
            # For each layer, compute Z using dot product by layer, while adding bias at each unit
            # Then pass it on to sigmoid function to get values in range [0, 1]
            Z[i-1] = np.dot(W[i-1], A[i-1]) + D[i-1]
            A[i] = self.sigmoid(Z[i-1])

    def backprop(self, A, D, W, Z, gradients):
        """
        Backward pass error derivatives
        Also, compute gradients in between.
        """
        for i in range(len(D) - 2, -1, -1):
            D[i] = np.multiply(np.dot(W[i+1].T, D[i+1]), self.sigmoid_prime(Z[i]))
            gradients[i] += np.outer(A[i], D[i]).T

        # Putting it together in stochastic gradient descent
    def SGD(self, X, Y, sizes, alpha=0.05, batch=10, epochs=1, debug=False):
        if debug:
            costVector = []
        (A, W, Z, D, gradients) = self.init_caches()
        m = len(Y)
        T = np.arange(m)
        for k in range(epochs):
            np.random.shuffle(T)
            k = 0
            while k < m:
                # TODO Test if gradients has to re-initialized for every batch.
                gradients = []
                for i in range(len(sizes) - 1):
                    gradients.append(np.zeros([sizes[i+1], sizes[i]]))
                for i in range(batch):
                    # Set A
                    A[0] = X[T[k+i]]
                    # Forward pass through network
                    self.forward_pass(A, D, W, Z)
                    # Do final layer's error
                    L = len(A) - 1
                    D[L-1] = A[L] - Y[T[k+i]]
                    # gradients for last layer
                    gradients[L-1] += np.outer(A[L-1], D[L-1]).T
                    # Back propagate errors.
                    self.backprop(A, D, W, Z, gradients)
                # Update weights & biases with new gradients.
                for j in range(len(W)):
                    W[j] -= (alpha*gradients[j])/batch
                    D[j] -= (alpha*D[j])/batch
                k += batch

                new_cost = self.costFunction(A, D, W, Z, X, Y)
                print ("loss: {0}".format(new_cost))
                if debug:
                    costVector.append(new_cost)

        if debug:
            return (W, D, costVector)
        else:
            return (W, D)


    # TODO Move to common utils
    def costFunction(self, A, D, W, Z, X, Y):
        res = 0
        m = len(Y)
        for i in range(m):
            # Compute activations by layer for each data point in X.
            A[0] = X[i]
            self.forward_pass(A, D, W, Z)
            L = len(A) - 1
            if A[L] > 0.5:
                t = 1
            else:
                t = 0
            res += np.sum((A[len(A) - 1] - Y[i])**2)

        return float(res/(2*m))

    def sigmoid(self, X):
        return 1/(1 + np.exp(np.negative(X)))

    def sigmoid_prime(self, X):
        sig = self.sigmoid(X)
        return sig*(1 - sig)

    def fit_SGD(self, X, Y,  alpha=0.001, batch=10, epochs=1, debug=False):
        (m,n) = X.shape
        self.__m__ = m
        self.__n__ = n
        self.__alpha__ = alpha
        W = np.zeros(n)

        if debug:
            (self.weights, self.biases, self.costVector) = self.SGD(X, Y, self.size, alpha=alpha, batch=batch, epochs=epochs, debug=True)
        else:
            (self.weights, self.biases) = self.SGD(X, Y, self.size, alpha=alpha, batch=batch, epochs=epochs, debug=False)

    def test(self, Xtest):
        if self.weights is not None:
            (m, n) = Xtest.shape
            assert n == self.__n__
            res = []
            A = []
            sizes = self.size
            for i in sizes:
                A.append(np.zeros(i))

            Z = []
            for i in range(1, len(sizes)):
                Z.append(np.zeros(sizes[i]))

            L = len(A) - 1
            for i in range(m):
                A[0] = Xtest[i]
                self.forward_pass(A, self.biases, self.weights, Z)

                if A[L] > 0.5:
                    res.append(1)
                else:
                    res.append(0)
            return res
        else:
            raise ValueError("Please fit the model first.")
