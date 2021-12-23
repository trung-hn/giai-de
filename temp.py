# %%
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
from random import randint
import collections
import functools
import time


def timer(func):
    """Print the runtime of the decorated function"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()    # 1
        value = func(*args, **kwargs)
        end_time = time.perf_counter()      # 2
        run_time = end_time - start_time    # 3
        print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return value
    return wrapper_timer


# %%


minimum_AB = 3000
skip = 0
AB_per_HR = 5
top = 5
get_a = 1
given_name = name = 0
batting_df = pd.read_csv(
    r"D:\OneDrive\Education\Drexel\2018 - 2019\Summer\CS 571\Assignments\Batting.csv")
people_df = pd.read_csv(
    r"D:\OneDrive\Education\Drexel\2018 - 2019\Summer\CS 571\Assignments\People.csv")

#people_df = people_df[["playerID", "nameFirst", "nameLast"]]
#batting_df = batting_df[["playerID", "AB", "HR", "H"]]
# %%
# combine values with same column
batting_df = batting_df.groupby(batting_df['playerID']).aggregate("sum")

# Merge 2 df
merge_df = batting_df.merge(
    people_df, left_on=batting_df.index, right_on="playerID")

# Create new columns
merge_df["AB/HR"] = merge_df["AB"]/merge_df["HR"]
merge_df["H/AB"] = merge_df["H"]/merge_df["AB"]
merge_df["First_Last"] = merge_df["nameFirst"] + merge_df["nameLast"]

merge_df = merge_df[["playerID", "First_Last",
                     "nameGiven", "AB/HR", "H/AB", "AB"]]

# %%
if given_name:
    merge_df = merge_df.rename(columns={"nameGiven": "displayName"})
elif name:
    merge_df = merge_df.rename(columns={"First_Last": "displayName"})
else:
    merge_df = merge_df.rename(columns={"playerID": "displayName"})

if get_a:
    temp = merge_df[merge_df["AB"] >= minimum_AB].sort_values(
        "AB/HR", ascending=True)
    for index, row in temp.iloc[skip: skip + top].iterrows():
        print(row["displayName"], row["AB/HR"])
    if skip+top > len(temp.index):
        print(
            "\nSkip + stop > available values. {} rows are printed".format(len(temp.index)))
else:  # get_b
    temp = merge_df[merge_df["AB"] >= minimum_AB].sort_values(
        "H/AB", ascending=False)
    ans = temp.iloc[skip: skip + top][["H/AB"]]
    for index, row in ans.iterrows():
        print(index, row[0])
    if skip+top > len(ans.index):
        print(
            "\nSkip + stop > available values. {} rows are printed".format(len(ans.index)))
# %%

data = np.array([[4, 0,	0,	0,	0,	0], [0,	181,	0,	0,	0,	0], [0,	0,	316,	0,	0,	0], [
                0,	0,	0,	8,	0,	0], [0,	0,	0,	0,	0,	0], [0,	0,	2,	0,	0,	13]])

fig, ax = plot_confusion_matrix(conf_mat=data)
plt.show()
# %%
L = [1, 2, 3]


def f(L, sofar=[]):
    if not L:
        print(sofar)
        return
    for i, num in enumerate(L):
        f(L[:i] + L[i+1:], sofar + [num])


f([3, 4, 1, 2])
f(["A", "B", "C", "D"])

# %%
i = 0
rv = ""
while i < 100:
    rv += str(i)[0]
    i += 1
    if i % 10 == 0:
        rv += "\n"

print(rv)


# %%
n = int(input("Enter Tree Level: "))
tree = (f'{"*" * (2 * j + 1) : ^{n * 2 + 3}}' for i in range(n)
        for j in range(i + 3))
print(*tree, sep="\n")


# %%
path = r'D:\Workspace\leetcode-solutions\src\7.reverse-integer.py'

with open(path, 'r') as f:
    rv = f.read()
    print(rv)
    print(list(rv))
    print(len(rv))


with open(path, 'r') as f:
    for line in f:

        # %%

        # 4
lst = []
for _ in range(3):
    lst.append(int(input("Enter Number: ")))
print(max(lst))

# 1
lst = []
while True:
    lst.append(int(input("Enter Number: ")))
    if 1 <= lst[-1] <= 10:
        break
print(sum(lst))

# %%
empoyees = [(11,  34, 78,  5,  11, 56),
            (12,  31, 98,  7,  34, 78),
            (13,  16, 11,  11, 56, 41),
            (89,  41, 12,  12, 78, 78)]
# Create a DataFrame object
df = pd.DataFrame(empoyees,
                  columns=['A', 'B', 'C', 'D', 'E', 'F'])

# %%

f0 = open(r"Desktop\MAXGIF.INP", "r")
f1 = open(r"Desktop\MAXGIF.OUT", "w")
n = int(input())
a = list(map(int, f0.readline().split()))
m = 0
for i in range(1, n):
    if m < a[i-1]+a[i]:
        m = a[i-1]+a[i]
# f1.write(m)
print(m, file=f1)
f0.close()
f1.close()
# %%

with open(r"Desktop\MAXGIF.txt", "r") as f:
    n = int(f.readline())
    a = list(map(int, f.readline().split()))

m = 0
for i in range(n):
    m = max(m, a[i - 1] + a[i])

with open(r"Desktop\MAXGIF.OUT", "w") as f:
    f.write(str(m))
# %%
f = open(r"Desktop\MAXGIF.INP", "r")
n = int(f.readline())
a = list(map(int, f.readline().split()))
f.close()

m = 0
for i in range(n):
    m = max(m, a[i - 1] + a[i])

f = open(r"Desktop\MAXGIF.OUT", "w")
f.write(str(m))
# hoac la print(m, file=f)
f.close()

# %%


def calculate(n):
    ans = 0
    for num in range(2, n + 1):
        diff = num - 1
        seen = set()
        flag = True

        # Check for each starting point
        for i in range(1, n + 1):
            if i in seen:
                continue
            if i + diff in seen:
                flag = False
            seen.add(i)
            seen.add(i + diff)
        if flag and len(seen) == n:
            # print(diff)
            ans += 1
    return ans


def calculate2(n):
    ans = 0
    for i in range(1, int(n ** 0.5 + 1)):
        if n % i == 0:
            r = n // i
            ans += (i % 2 == 0) + (r % 2 == 0) - (r == i)
    return ans


for i in range(1, 20):
    print(i, i * 2,  calculate(i * 2), calculate2(i * 2))

print(all(calculate(i * 2) == calculate2(i * 2) for i in range(100)))

# %%
mat = [[0] * abs(N) for _ in range(abs(N))]
for i in range(abs(N)):
    mat[i][i if N > 0 else ~i] = 1
print(mat)

# %%
with open(r"Desktop\MAXGIF.INP", "r") as f:
    arr = [int(val) for val in f.readlines()]


def cc(n):
    ans = 0
    for num in range(2, n + 1):
        diff = num - 1
        seen = set()
        flag = True

        # Check for each starting point
        for i in range(1, n + 1):
            if i in seen:
                continue
            if i + diff in seen:
                flag = False
            seen.add(i)
            seen.add(i + diff)
        if flag and len(seen) == n:
            # print(diff)
            ans += 1
    return ans


with open(r"Desktop\CACHCHIA.OUT", "w") as f:
    for num in arr:
        f.write(str(cc(num * 2)) + "\n")
# %%
N = 10
for i in range(N):
    j = N - 1 - i
    print(" " * (j * 2), end="")
    print("*" * (i * 2 + 1))


# %% Bai 2

def get_divisors(n):
    rv = set()
    for i in range(2, int(n ** 0.5 + 1)):
        if n % i == 0:
            rv |= {i, n / i}
    return rv


def main(arr):
    seen = set(arr)
    ans = []
    max_cnt = 0
    for num in arr:

        # Overlapped set
        overlap = seen & get_divisors(num)
        cnt = len(overlap)

        # Edge case
        if cnt == 0 or cnt < max_cnt:
            continue

        # Answer
        if cnt > max_cnt:
            ans = [num]
        elif cnt == max_cnt:
            ans.append(num)

        max_cnt = max(max_cnt, cnt)
    return sorted(ans) or "KHONG"


print(main([7, 2, 28, 5, 8, 4, 20, 10, 12, 15]))
print(main([7, 3, 2, 11, 13]))

# %% Bai 4


def main(N, arr):
    ans = [0] * N
    for val in arr:
        ans[val] = 1
    return sum(ans)


print(main(11, [1, 2, 3, 4, 5, 1, 2, 1, 2, 7, 5]))


# %% Bai 1

def dfs(fib, N, ptr=0, sofar=[]):
    if N == 0:
        return sofar
    for i, val in enumerate(fib[ptr:], ptr):
        if rv := dfs(fib, N - val, i + 1, sofar + [val]):
            return rv


def main(N):
    fib = [1, 2]
    while fib[-1] < N:
        fib.append(fib[-1] + fib[-2])

    fib.reverse()
    return dfs(fib, N)


print(main(16))
print(main(33))

# %% Bài 3


def main(arr):

    multi = False
    longest = 0
    curr = [arr[0]]
    for n1, n2 in zip(arr, arr[1:]):

        # Same sign numbers
        if n1 >= 0 and n2 >= 0 or n1 < 0 and n2 < 0:
            curr.append(n2)
        else:
            curr = [n2]

        # Assign
        if len(curr) > longest:
            multi = False
            longest = len(curr)
        elif len(curr) == longest:
            multi = True

    if multi:
        print("co nhieu")
    return curr


print(main([-7, 4, -8, 9, 25, 17]))
print(main([6, 3, 8, 16, -5, 8, -6, 4, -9, -4, -1, -5]))

# %%
a = 2
b = 5
for i in range(a, b+1):
    print(f"i = {i}")
    for j in range(i+1, b+1):
        print(f"   j = {j}")


# %%


def random_x():
    n = randint(5, 20)
    while n <= 16:
        print("x" * n)
        n = randint(5, 20)


random_x()

# %% DISKS

"""
dfs(14, 0, []):
    dfs(14 - 9, 0 + 1, [9]):
        dfs(5 - 4, 4 + 1, [9, 4]):
            ...
        dfs(5 - 3, 5 + 1, [9, 3]):
            dfs(2 - 2, 7 + 1, [9, 3, 2]):
                target == 0, check available => reduce availalle => add to ans
                
    dfs(14 - 8, 1 + 1, [8]):
        dfs(6 - 6, 3 + 1, [8, 6]):
            target == 0, check available => reduce availalle => add to ans
    
    dfs(14 - 7, 3 + 1, [7]):
        dfs(7 - 4, 4 + 1, [7, 4]):
            dfs(3 - 3, 5 + 1, [7, 4, 3]):
                target == 0, check available => reduce availalle => add to ans
"""


def get_counter(nums):
    freqs = [0] * (max(nums) + 1)
    for num in nums:
        freqs[num] += 1
    print(freqs)
    return freqs


def main(N, M, nums):

    nums.sort(reverse=True)
    print(nums)
    freqs = get_counter(nums)

    # visited = collections.defaultdict(list)

    rv = []

    def dfs(target=14, ptr=0, arr=[]):

        # Base case
        if target == 0:

            # Check if freq is available ?
            if all(freqs[num] > 0 for num in arr):
            if available, reduce freq
            for num in arr:

                freqs[num] -= 1
            # Append to ans
            rv.append(arr)
            return

        # Recursive Case
        for i in range(ptr, len(nums)):
            num = nums[i]
            if num <= target:
                dfs(target - num, i + 1, arr + [num])

    dfs(target=M)
    return len(rv), rv


print(main(8, 14, [9, 7, 4, 3, 3, 2, 8, 6]))

# %%


def transpose(mat):
    R, C = len(mat), len(mat[0])
    temp = [[0] * R for _ in range(C)]

    for r in range(R):
        for c in range(C):
            temp[c][r] = mat[r][c]

    return temp


def mirror(mat):
    temp = []
    for row in mat:
        temp_row = []
        for i in range(len(row)):
            temp_row.append(row[len(row) - 1 - i])
        temp.append(temp_row)
    return temp


# transpose([[1,2,3], [4,5,6]])
rv = transpose(mirror(transpose(mirror([[1, 2, 3], [4, 5, 6]]))))
print(*rv, sep="\n")


# %% DUONG HAM
"""
dfs(1): visited = {1}
    dfs(3): visited = {1, 3}
        dfs(1): <=    
    dfs(5): visited = {1, 3, 5}
        dfs(1): <=
    dfs(6): visited = {1, 3, 5, 6}
        dfs(1): <=
    <=
"""


def main(N, connections):

    neis = collections.defaultdict(list)

    for start, end in connections:
        neis[start-1].append(end-1)
        neis[end-1].append(start-1)

    print(neis)

    visited = set()

    def dfs(node):
        if node in visited:
            return
        visited.add(node)
        for nei in neis[node]:
            dfs(nei)

    ans = 0
    for i in range(N):
        if i not in visited:
            dfs(i)
            ans += 1
    return ans - 1


print(main(9, [[1, 3], [1, 5], [1, 6], [2, 7], [4, 8], [8, 9]]))

# %% DU TRU NUOC


def main(N, arr):

    water = 0
    storages = []
    for leftover, storage in arr:
        water += leftover
        storages.append(storage)

    storages.sort()

    ans = N
    for storage in storages:
        water -= storage
        if water >= 0:
            ans -= 1
        else:
            break
    return ans


print(main(4, [[0, 1], [3, 5], [0, 2], [1, 2]]))


# %%


@timer
def main(R, C, mat):
    def get_max_area(row):
        row.append(-1)
        ans = 0
        stack = [-1]
        for i, val in enumerate(row):
            # Monotonic Stack
            while stack and row[stack[-1]] > val:
                H = row[stack.pop()]
                W = i - stack[-1] - 1
                ans = max(ans, H * W)
            stack.append(i)
        return ans

    def maximum_rectangle(mat):
        ans = 0
        for r in range(R):
            for c in range(C):
                if mat[r][c]:
                    # DP
                    mat[r][c] += mat[r - 1][c]
            ans = max(ans, get_max_area(mat[r]))
        return ans

    def recreate_mat(mat, number):
        return [[val == number for val in row] for row in mat]

    seen = set(val for row in mat for val in row)
    ans = 0
    for num in seen:
        new_mat = recreate_mat(mat, num)
        ans = max(ans, maximum_rectangle(new_mat))
    return ans


print(main(4, 5, [[1, 2, 1, 3, 2], [3, 2, 2, 2, 1, ],
      [1, 2, 2, 2, 1], [1, 3, 3, 3, 2]]))

arr = [random.choices(range(2), k=100) for _ in range(66000)]
print(main(len(arr), len(arr[0]), arr))
