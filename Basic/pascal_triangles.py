# %%
def pascal_triangle(N=5):

    # Init
    nums = [[0] * (N * 2 - 1) for _ in range(N)]

    # Pre-filled
    nums[0][N - 1] = nums[-1][0] = nums[-1][-1] = 1
    R, C = len(nums), len(nums[0])

    # Fill in
    for r in range(1, R):
        for c in range(1, C - 1):
            total = nums[r - 1][c - 1] + nums[r - 1][c + 1]
            if total in (0, 1):
                nums[r][c] = total
            else:
                nums[r][c] = total + 1
    return nums


for row in zip(*pascal_triangle(5)):
    print("".join(f"{val if val else ' ':^5}" for val in row))
