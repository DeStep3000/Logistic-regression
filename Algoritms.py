import numpy as np
import time
import matplotlib.pyplot as plt


# sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# gradient descent algorithm
def gradient_descent(X, y, theta, alpha, lamda, iterations):
    m = len(y)
    for i in range(iterations):
        h = sigmoid(X.dot(theta))
        grad = (1 / m) * X.T.dot(h - y) + (lamda / m) * theta
        theta -= alpha * grad
    return theta


# Newton's method algorithm
def newtons_method(X, y, theta, lamda, iterations):
    m = len(y)
    for i in range(iterations):
        h = sigmoid(X.dot(theta))
        grad = (1 / m) * X.T.dot(h - y) + (lamda / m) * theta
        H = (1 / m) * X.T.dot(np.diag(h * (1 - h)).dot(X)) + (lamda / m) * np.eye(theta.shape[0])
        theta -= np.linalg.inv(H).dot(grad)
    return theta


# generate random dataset
def generate_dataset(n):
    X = np.random.rand(100, n)
    y = np.random.randint(2, size=100)
    return X, y


# measure algorithm running time
def measure_time(algorithm, X, y, theta, *args):
    start_time = time.time()
    theta2 = algorithm(X, y, theta, *args)
    end_time = time.time()
    print("\n", theta2)
    return end_time - start_time


# compare algorithms on different dataset sizes
results = []
for n in range(10, 101, 10):
    X, y = generate_dataset(n)
    X = np.insert(X, 0, 1, axis=1)
    theta = np.zeros(n + 1)
    gd_time = measure_time(gradient_descent, X, y, theta, 0.1, 0.1, 1000)
    nt_time = measure_time(newtons_method, X, y, theta, 0.1, 1000)
    results.append((n, gd_time, nt_time))

plt.plot([r[0] for r in results], [r[1] for r in results], label='Gradient Descent')
plt.plot([r[0] for r in results], [r[2] for r in results], label="Newton's Method")
plt.xlabel('Dataset size (n)')
plt.ylabel('Running Time (seconds)')
plt.title('Comparison of Algorithms')
plt.legend()
plt.show()
