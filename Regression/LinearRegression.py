import numpy as np

def prediction(theta, X):
    return float(X.dot(theta))              # Find theta-transpose-X

def gradientDescent(theta, X, Y, alpha=0.01, num_iters=1000):
    (m, n) = X.shape

    for k in range(num_iters):              # For each iteration
        for j in range(n):                  # For each theta(i) or x(i) in X[i]
            gradient = 0
            for i in range(m):              # For each data point (X[i], y[i])
                pred = prediction(theta, X[i])
                gradient += (pred - Y[i])*X[i][j]
            theta[j] -= float(alpha*(gradient/m))

    return theta


if __name__ == "__main__":
    X = np.random.randint(0, 20, size=(20, 4))
    y = np.random.randint(0, 2, size=20)
    (m, n) = X.shape
    init_theta = np.zeros((n, 1))
    alpha = 0.1

    # Find theta values
    theta = gradientDescent(init_theta, X, y, alpha)

    print ("theta: ", theta)                # Final theta or weights.

    # Testing data
    X_test = np.random.randint(0, 20, size=(1, 4))
    print ( "Linear Regression predicted value: ", prediction(theta, X_test) )
