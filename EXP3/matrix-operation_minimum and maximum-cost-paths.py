# %%
def min_flips_to_make_sum_non_negative(arr):
    total = sum(arr)
    n = len(arr)
    dp = [[float('inf')] * (total + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = 0

    for i in range(1, n + 1):
        for j in range(1, total + 1):
            if arr[i - 1] <= j:
                dp[i][j] = min(dp[i - 1][j], dp[i - 1][j - arr[i - 1]] + 1)
            else:
                dp[i][j] = dp[i - 1][j]

    result = float('inf')
    for j in range(total // 2 + 1):
        result = min(result, dp[n][j])

    return result

arr = [14, 10, 4]
print(min_flips_to_make_sum_non_negative(arr))  # Output: 1

# %%
def max_cost_path(grid):
    m, n = len(grid), len(grid[0])
    dp = [[0] * n for _ in range(m)]

    for i in range(m - 1, -1, -1):
        for j in range(n):
            if i == m - 1 and j == 0:
                dp[i][j] = grid[i][j]
            elif i == m - 1:
                dp[i][j] = dp[i][j - 1] + grid[i][j]
            elif j == 0:
                dp[i][j] = dp[i + 1][j] + grid[i][j]
            else:
                dp[i][j] = max(dp[i][j - 1], dp[i + 1][j]) + grid[i][j]

    return dp[0][n - 1]

grid = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]
print(max_cost_path(grid))  # Output: 26

# %%
from itertools import combinations

def max_nPr_pair(arr):
    n = len(arr)
    max_nPr = -1
    result = ()

    for i, j in combinations(range(n), 2):
        nPr = arr[i] * arr[j]
        if nPr > max_nPr:
            max_nPr = nPr
            result = (arr[i], arr[j])

    return result

arr = [5, 2, 3, 4, 1]
print(max_nPr_pair(arr))  # Output: (5, 4)

# %%
def min_operations_to_sort(arr):
    n = len(arr)
    operations = 0

    for i in range(1, n):
        if arr[i] < arr[i - 1]:
            operations += arr[i - 1] - arr[i]
            arr[i] = arr[i - 1]

    return operations

arr1 = [1, 2, 1, 4, 3]
arr2 = [1, 2, 2, 100]

print(min_operations_to_sort(arr1))  # Output: 2
print(min_operations_to_sort(arr2))  # Output: 0

# %%


