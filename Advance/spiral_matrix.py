# %%
def rotate_m90(mat):
    return list(zip(*map(reversed, mat)))


def main(N):
    start = N**2 - 1
    mat = [[start]]
    while start > 0:
        C = len(mat)
        mat = rotate_m90(mat) + [range(start - C, start)]
        start -= C

    for row in reversed(mat):
        for val in row:
            print(f"{val:02}", end="\t")
        print()


main(5)

# %%


def main2(N):
    start = 1
    mat = [[start]]
    while start < N ** 2:
        C = len(mat)
        mat = rotate_m90(mat) + [tuple(range(start + C, start, - 1))]
        start += C
    print(*reversed(mat), sep="\n")


main2(4)
