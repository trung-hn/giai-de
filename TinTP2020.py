# %% Bai 4


def get_4d(r, c):
    return [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]


def main(start, end, arr):
    R, C = len(arr), len(arr[0])

    neis = collections.defaultdict(list)

    for r in range(R):
        for c in range(C):
            neis[arr[r][c]].append((r, c))

    q = [[*start, 0]]
    for r, c, d in q:
        if r == end[0] and c == end[1]:
            return d

        for x, y in set(neis[arr[r][c]] + get_4d(r, c)):

            if x == r and y == c:
                continue
            if 0 <= x < R and 0 <= y < C:
                q.append([x, y, d + 1])


print(main([0, 0], [4, 3], [[1, 2, 3, 4], [5, 0, 0, 6],
      [7, 0, 8, 6], [0, 0, 6, 0], [3, 4, 7, 9]]))


# %% Bai 1

# O(N)
def main(L=2, R=7):
    prefix = [L]
    for num in range(L + 1, R + 1):
        prefix.append(prefix[-1] + num)

    suffix = [R]
    for num in reversed(range(L, R)):
        suffix.append(suffix[-1] + num)
    suffix.reverse()

    min_sofar = float("inf")
    ans = 0
    for i, (s1, s2) in enumerate(zip(prefix, suffix[1:]), L):
        if abs(s1 - s2) < min_sofar:
            min_sofar = abs(s1 - s2)
            ans = i
    return ans


print(main())

# %% Bai 2


def main(N):
    if sum(map(int, str(N))) % 3:
        return -1
    chars = [c for c in str(N)]

    freqs = [0] * 10
    for c in chars:
        freqs[int(c)] += 1

    if freqs[0] == 0:
        return -1

    rv = ''
    for i in reversed(range(0, 10)):
        rv += freqs[i] * str(i)

    return int(rv)


print(main(1002963963))
print(main(1002))
print(main(1249857859))

# %% Bai 3


def main(N, M, arr):

    delta = collections.defaultdict(int)

    for start, end in arr:
        delta[start] += 1
        delta[end + 1] -= 1

        if start > end:
            delta[0] += 1

    rv = [0]
    for de in delta:
        rv.append(rv[-1] + de)

    rv = rv[1:-1]
    mx = max(rv)
    return mx, rv.count(mx)


print(main(5, 2, [[0, 4], [3, 1]]))
