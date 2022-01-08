# %% Phan tich so
import functools


def sol1(N):
    """
    Time and Space: O(2^N)
    Recursion
    """
    def dfs(total, prev=1):
        if total == 0:
            return 1
        return sum(dfs(total - i, i)
                   for i in range(prev, total + 1))
    return dfs(N)


def sol2(N):
    """
    Time and Space: O(N*N)
    Recursion with memoization
    """
    @functools.cache
    def dfs(total, prev=1):
        if total == 0:
            return 1
        return sum(dfs(total - i, i)
                   for i in range(prev, total + 1))
    return dfs(N)


def sol3(N):
    """
    Dynamic Programming
    Time and Space: O(N*N)
    Approach:
    We know:

    1 = 1

    2 = 1 + 1
      = 2

    3 = 1 + 2 = 1 + 1 + 1
              = 1 + 2
      = 3


    4 = 1 + 3 = 1 + 1 + 1 + 1
              = 1 + 1 + 2
              = 1 + 3
      = 2 + 2 = 2 + 2
      = 4

    5 = 1 + 4 = 1 + 1 + 1 + 1 + 1
              = 1 + 1 + 1 + 2
              = 1 + 1 + 3
              = 1 + 2 + 2
              = 1 + 4
      = 2 + 3 = 2 + 3
      = 5

    You can see, in order to create a sequence that does not repeat, it has to be non-decreasing.
    Thus, we can think of dp: 
    dp[i][j] is no of sequences start with j or more that i has.

    dp[4][4] = 1 (the sequence is itself)
    dp[4][3] = 1
    dp[4][2] = dp[2][2] + dp[4][3]
        sequence format: 2 ... (numbers >= 2, total is 2)
    dp[4][1] = dp[3][1] + dp[4][2]
        sequence format: 1 ... (numbers >= 1, total is 3)

    """
    dp = [[0] * N for _ in range(N + 1)]
    dp[1][1] = 1

    for i in range(2, N + 1):
        half = i // 2 + 1
        dp[i][half: i + 2] = [1] * half
        for j in reversed(range(half)):
            dp[i][j] = dp[i - j][j] + dp[i][j + 1]
    return dp[N][0]


N = 5
print(sol1(N))
print(sol2(N))
print(sol3(N))
