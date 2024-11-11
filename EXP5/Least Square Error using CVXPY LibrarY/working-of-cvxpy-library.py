import cvxpy as cp
import numpy as np

# Generate random data
np.random.seed(0)
m = 10  # Number of equations
n = 5   # Number of variables
A = np.random.randn(m, n)  # Random coefficient matrix
b = np.random.randn(m)      # Random right-hand side vector

# Define variables
x = cp.Variable(n)  # Variable vector of size n

# Define objective function: Minimize sum of squares of residuals
objective = cp.Minimize(cp.sum_squares(A @ x - b))

# Define constraint: Norm of x <= 1 (Euclidean norm)
constraints = [cp.norm(x) <= 1]

# Formulate and solve problem
problem = cp.Problem(objective, constraints)
problem.solve()

# Display results
print("Optimal value (minimum sum of squares):", problem.value)
print("Optimal solution x:", x.value)


