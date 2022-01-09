# %% Spiral Matrix
def rotate_90(mat):
    return list(map(reversed, zip(*mat)))


def main(N):
    num = N**2
    mat = [[num]]
    while num > 1:
        mat = [range(num - len(mat), num)] + rotate_90(mat)
        num -= len(mat[0])

    for row in mat:
        print(*row)


main(5)
# 1 2 3 4 5
# 16 17 18 19 6
# 15 24 25 20 7
# 14 23 22 21 8
# 13 12 11 10 9
