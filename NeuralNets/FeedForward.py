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
        self.__m__ = None
        self.__n__ = None
        self.__alpha__ = None
        self.__gradientStoppingDiff__ = None
        self.costVector = None
        # TODO Create blank network with weights, biases, activations etc.

    def init_caches(self, sizes):
        A = []
        for i in sizes:
            A.append(np.zeros(i))

        Z = []
        for i in range(1, len(sizes)):
            Z.append(np.zeros(sizes[i]))

        # Initialize Error cache
        D = Z

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
            A[i] = sigmoid(Z[i-1])

    def backprop(self, A, D, W, Z):
        """
        Backward pass error derivatives
        Also, compute gradients in between.
        """
        for i in range(len(D) - 2, -1, -1):
            D[i] = np.multiply(np.dot(W[i+1].T, D[i+1]), sigmoid_prime(Z[i]))
            gradients[i] += np.outer(A[i], D[i]).T

    # Putting it together in stochastic gradient descent
    def SGD(self, X, Y, sizes, alpha=0.05, batch=10, debug=False):
        if debug:
            costVector = []

        (A, W, Z, D, gradients) = self.init_caches(sizes)
        m = len(Y)
        T = np.arange(m)
        np.random.shuffle(T)
        k = 0
        while k < m:
            for i in range(batch):
                # Set A
                A[0] = X[T[k+i]]
                # Forward pass through network
                forward_pass(A, D, W, Z)
                # Do final layer's error
                L = len(A) - 1
                D[L-1] = A[L] - Y[T[k+i]]
                # Back propagate errors.
                backprop(A, D, W, Z)
            # Update weights & biases with new gradients.
            for j in range(len(W)):
                W[j] -= (alpha*gradients[j])/batch
                D[j] -= (alpha*D[j])/batch
            k += batch

            new_cost = costFunction(A, D, W, Z, Y)
            print ("loss: {0}".format(new_cost))
            if debug:
                costVector.append(new_cost)

        if debug:
                return (W, costVector)
        else:
            return W

    # TODO Move to common utils
    def costFunction(A, D, W, Z, Y):
        """
        L2 cost function
        """
        res = 0
        m = len(Y)
        for i in range(m):
            # Compute activations by layer for each data point in X.
            forward_pass(A, D, W, Z)
            res += np.sum((A[len(A) - 1] - Y[1])**2)

        return float(res/(2*m))

    def sigmoid(X):
        return 1/(1 + np.exp(np.negative(X)))

    def sigmoid_prime(X):
        sig = sigmoid(X)
        return sig*(1 - sig)
