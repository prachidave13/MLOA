import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

# Define loss function
def loss_fn(X, Y, beta):
    return cp.norm(X @ beta - Y, p=2)**2 / X.shape[0]

# Define regularizer (L1 norm)
def regularizer(beta):
    return cp.norm(beta, p=1)

# Define objective function
def objective_fn(X, Y, beta, lambd):
    return loss_fn(X, Y, beta) + lambd * regularizer(beta)

# Define mean squared error function
def mse(X, Y, beta):
    return np.linalg.norm(X.dot(beta) - Y)**2 / len(Y)

# Generate synthetic data
def generate_data(m=100, n=20, sigma=5, density=0.2):
    np.random.seed(1)
    beta_star = np.random.randn(n)
    idxs = np.random.choice(range(n), int((1-density)*n), replace=False)
    beta_star[idxs] = 0
    X = np.random.randn(m, n)
    Y = X.dot(beta_star) + np.random.normal(0, sigma, size=m)
    return X, Y, beta_star

# Set parameters for data generation
m = 100
n = 20
sigma = 5
density = 0.2

# Generate data
X, Y, _ = generate_data(m, n, sigma)
X_train = X[:50, :]
Y_train = Y[:50]
X_test = X[50:, :]
Y_test = Y[50:]

# Define variables
beta = cp.Variable(n)
lambd = cp.Parameter(nonneg=True)
lambd_values = np.logspace(-2, 3, 50)
train_errors = []
