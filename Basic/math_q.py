# %%
# find number of a, n such that a/m - b/n = (a-b)/(m-n)


def main(b, n):
    pos_m = [n * (1 + (1 - a / b) ** 0.5) for a in range(b)]
    neg_m = [n * (1 - (1 - a / b) ** 0.5) for a in range(b)]
    return sum(m > 0 and abs(m - round(m)) < 10e-6 for m in pos_m + neg_m)


main(9, 12)
