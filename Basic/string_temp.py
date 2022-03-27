# %%

def main1(S, N):
    N %= len(S)
    return S[N:] + S[:N]


print(main1("0123456789", 1))    # 123456789
print(main1("0123456789", 11))   # 123456789
print(main1("0123456789", 101))  # 123456789

# %%


def convert_26_to_10(s):
    res = 0
    for i, c in enumerate(s):
        res += (ord(c) - 64) * 26 ** (len(s) - i - 1)
    return res


def convert_10_to_26(N):
    res = []
    while N > 26:
        r = (N - 1) % 26 + 1
        res.append(r + 64)
        N = N // 26 if N % 26 else N // 26 - 1
    res.append(N + 64)
    return ''.join(reversed([chr(val) for val in res]))


print(convert_26_to_10("A"))
print(convert_26_to_10("AA"))
print(convert_26_to_10("ZZ"))
print(convert_10_to_26(1))
print(convert_10_to_26(27))
print(convert_10_to_26(702))


for i in range(100000):
    if i != convert_26_to_10(convert_10_to_26(i)):
        print(i, convert_10_to_26(i))
