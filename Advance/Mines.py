# Answer for Mines:
def main(arr, N, M):
    N *= 2
    dp = [1, 0, 1] + [0] * N
    for L in range(4, N + 2, 2):
        for i in range(2, L + 1, 2):
            dp[L] += dp[i - 2] * dp[L - i]

    def dfs(left=0, right=None):
        """
        Left is when a person comes in, right is where that same person comes out
        right is None when we are not sure when that person comes out (require checking with M)
        """
        if left == N:
            return 1
        if right is not None:
            return dp[right - left + 1]

        ans = 0
        # Try all `right` until we find invalid `right`
        for right in range(left + 1, N + 1, 2):
            if arr[left] + M < arr[right]:
                break
            # There must be people come in and out between `left` and `right`
            # There alsto must be people come in and out after `right``
            ans += dfs(left + 1, right - 1) * dfs(right + 1)
        return ans
    return dfs()


print(main([1, 2, 3, 7, 9, 10], 3, 6))
print(main(list(range(10)), 5, 10))
