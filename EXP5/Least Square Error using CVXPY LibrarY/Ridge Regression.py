import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

def loss_fn(X, Y, beta):
    return cp.norm(X @ beta - Y, p=2)**2 / X.shape[0]

def regularizer(beta):
    return cp.norm(beta, p=2)**2

def objective_fn(X, Y, beta, lambd):
    return loss_fn(X, Y, beta) + lambd * regularizer(beta)

def mse(X, Y, beta):
    return np.linalg.norm(X.dot(beta) - Y)**2 / len(Y)

def generate_data(m=100, n=20, sigma=5):
    np.random.seed(1)
    beta_star = np.random.randn(n)
    X = np.random.randn(m, n)
    Y = X.dot(beta_star) + np.random.normal(0, sigma, size=m)
    return X, Y

m = 100
n = 20
sigma = 5

X, Y = generate_data(m, n, sigma)
X_train = X[:50, :]
Y_train = Y[:50]
X_test = X[50:, :]
Y_test = Y[50:]

beta = cp.Variable(n)
lambd = cp.Parameter(nonneg=True)
lambd_values = np.logspace(-2, 3, 50)
train_errors = []
test_errors = []
beta_values = []

for v in lambd_values:
    lambd.value = v
    problem = cp.Problem(cp.Minimize(objective_fn(X_train, Y_train, beta, lambd)))
    problem.solve()
    train_errors.append(mse(X_train, Y_train, beta.value))
    test_errors.append(mse(X_test, Y_test, beta.value))
    beta_values.append(beta.value)

def plot_train_test_errors(train_errors, test_errors, lambd_values):
    plt.plot(lambd_values, train_errors, label="Train error")
    plt.plot(lambd_values, test_errors, label="Test error")
    plt.xscale("log")
    plt.legend(loc="upper left")
    plt.xlabel(r"$\lambda$", fontsize=16)
    plt.title("Mean Squared Error (MSE)")
    plt.show()

plot_train_test_errors(train_errors, test_errors, lambd_values)


def plot_regularization_path(lambd_values, beta_values):
    num_coeffs = beta_values[0].shape[0]
    for i in range(num_coeffs):
        plt.plot(lambd_values, [wi[i] for wi in beta_values])
    plt.xlabel(r"$\lambda$", fontsize=16)
    plt.xscale("log")
    plt.title("Regularization Path")
    plt.show()

plot_regularization_path(lambd_values, beta_values)

