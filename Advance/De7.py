# %% DU TRU NUOC

# Time: O(NlogN). Space: O(N)
import itertools


def prob1(N, arr):
    total = 0
    storages = []
    for water, size in arr:
        total += water
        storages.append(size)

    storages.sort()
    print(storages)
    for i, storage in enumerate(storages):
        if total >= storage:
            total -= storage
        else:
            return len(storages) - i


prob1(4, [[0, 1], [3, 5], [0, 2], [1, 2]])
prob1(5, [[0, 1], [0, 2], [0, 3], [4, 4], [5, 5]])

# %%


def prob2(N, connections):

    neis = [[] for _ in range(N + 1)]
    for start, end in connections:
        neis[start].append(end)
        neis[end].append(start)

    print(neis)

    def dfs(island, visited):
        if visited[island]:
            return
        visited[island] = 1
        for nei in neis[island]:
            dfs(nei, visited=visited)

    ans = 0
    visited = [0] * (N + 1)
    for island in range(1, N + 1):
        if visited[island]:
            continue
        dfs(island, visited=visited)
        ans += 1
    return ans - 1


prob2(9, [[1, 3], [1, 5], [1, 6], [2, 7], [4, 8], [8, 9]])

# %%


def prob3(R, C, arr):  # O(D * R * C)

    # Monotonic Stack (ASC)
    def find_max_area(row):  # O(R)
        row.append(-1)
        stack = [-1]
        max_area = 0
        for i, val in enumerate(row):
            while stack and row[stack[-1]] > val:
                height = row[stack.pop()]
                left = stack[-1]
                right = i
                area = height * (right - left - 1)
                max_area = max(max_area, area)
            stack.append(val)
        return max_area

    # dp
    def find_max_rectangle(R, C, mat):  # O(R*C)
        max_area = 0
        for r in range(1, R):  # O(R)
            for c in range(C):  # O(C)
                if mat[r][c]:
                    mat[r][c] += mat[r - 1][c]
            max_area = max(max_area, find_max_area(mat[r]))  # O(C)
        return max_area

    def get_level(depth, arr):
        return [[[0, 1][val == depth] for val in row]for row in arr]

    depths = set(itertools.chain(*arr))
    ans = 0
    for depth in depths:  # O(D)
        mat = get_level(depth, arr)  # O(R * C)
        ans = max(ans, find_max_rectangle(R, C, mat))  # O(R*C)
    return ans


prob3(4, 5, [[1, 2, 1, 3, 2], [3, 2, 2, 2, 1],
      [1, 2, 2, 2, 1], [1, 3, 3, 3, 2]])

"""
[6, 2, 5, 4, 5, 1, 6, -1]
stack = [-1, 1, 3]
max_area = 12

i=0, val=6:
    stack.append(0)

i=1, val=2:
    right (i=0, val=6) = 1
    left (i=0, val=6) = -1
    W = 1 - (-1) -1 = 1
    H = 6
    area = 6
    stack.append(1)

i=2, val=5:
    stack.append(2)

i=3, val=4:
    right (i=2, val=5) = 3
    left (i=2, val=5) = 1 
    W = 3 - 1 - 1 = 1
    H = 5
    area = 5
    stack.append(3)

i=4, val=5:
    stack.append(4)

i=5, val=1:
    right (i=4, val 5) = 5
    left (i=4, val 5) = 3
    W = 5 - 3 - 1 =1
    H = 5
    area = 5 * 1

    right (i=3,val=4) = 5
    left (i=3,val=4) = 1
    W = 5 - 1 - 1 =3
    H = 4
    area = 12

    right (i=1, val=2) = 5
    left (i=1, val=2) = -1
    W = 5 - (-1) - 1 = 5
    H = 2
    area = 10
"""
