# %%
import enum
import itertools
import os
import requests
from matplotlib.figure import Figure
from matplotlib.backend_bases import key_press_handler
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
import tkinter
from types import new_class
from sympy import *
import math
import random
import string
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
import collections
import time
import functools


def timer(func):
    """Print the runtime of the decorated function"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        print(f"Finished {func.__name__!r} in {run_time:.6f} secs")
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


def main(N, M, nums):
    nums.sort(reverse=True)
    print(nums)
    freqs = collections.Counter(nums)
    rv = []

    def dfs(target=14, ptr=0, arr=[]):

        # Base case
        if target == 0:

            # Check if freq is available ?
            if all(freqs[num] > 0 for num in arr):
                # if available, reduce freq
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


@timer
def write_disk(N, M, nums):
    nums.sort(reverse=True)
    rv = []

    def get_best_combo(target=14, ptr=0, combo=[]):
        # Base case
        rv_target = target
        rv_combo = combo
        # Recursive case
        for i in range(ptr, len(nums)):
            num = nums[i]
            if num <= target and freqs[num]:
                freqs[num] -= 1
                c_target, c_combo = get_best_combo(
                    target - num, i + 1, combo + [num])
                if c_target < rv_target:
                    rv_target = c_target
                    rv_combo = c_combo
                freqs[num] += 1

        return rv_target, rv_combo

    freqs = collections.Counter(nums)
    for num in nums:
        if freqs[num]:
            freqs[num] -= 1
            _, combo = get_best_combo(M - num)
            for val in combo:
                freqs[val] -= 1
            rv.append([num] + combo)
    return rv


print(write_disk(8, 14, [9, 7, 4, 3, 3, 2, 8, 6]))
print(write_disk(2, 14, [10, 3]))
print(write_disk(10, 1000, [3, 7, 6, 8, 8, 7, 6, 4, 4, 7]))

arr = random.choices(range(1, 10), k=30)
print(write_disk(len(arr), 20, arr))

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


# %%


def duplicate(arr):
    return [duplicate(val) if isinstance(val, list) else val for val in arr]


x = [1, 3, 6, [18]]
y = duplicate(x)
# y = x[:]
x[3][0] = 15
x[1] = 12
print(x)
print(y)


# %%

def main(N):
    mat = [[val for val in range(i, i*N + 1, i)] for i in range(1, N + 1)]
    for i in range(2, N):
        start = 4 if i == (N - 1) else 2
        mat[i][max(N - i, 2):] = range(start, i * 2 + 1, 2)
    return mat

# output:
# [1, 2, 3, 4, 5, 6]
# [2, 4, 6, 8, 10, 12]
# [3, 6, 9, 12, 2, 4]
# [4, 8, 12, 2, 4, 6]
# [5, 10, 2, 4, 6, 8]
# [6, 12, 4, 6, 8, 10]


print(*main(6), sep="\n")

# %%


def sort_num(num):
    freqs = [0] * 10
    while num:
        freqs[num % 10] += 1
        num //= 10
    return (str(i) * freqs[i] for i in range(1, 10))


def main(N):
    a = 1
    for _ in range(N - 1):
        a = int("".join(sort_num(a * 2)))
    return a


print(main(7))

# %%
# import sympy


x, y = symbols('x y')
f = 3*y**2 - 2*y**3 - 3*x**2 + 6*x*y
f = x*y + 4
diff_x = diff(f, x)
diff_y = diff(f, y)
print(f)
print(diff_x)
print(diff_y)
res = solve((diff_x, diff_y), x, y)
print(res)
if isinstance(res, dict):
    res = [[res[x], res[x]]]
print(res)
for xm, ym in res:
    diff_xx = diff(diff_x, x, 1)
    print(diff_xx)
    diff_xy = diff(diff_x, y, 1)
    diff_yy = diff(diff_y, y, 1)
    d = diff_xx * diff_yy - (diff_xy) ** 2
    A = d.subs([(x, xm), (y, ym)])
    fxxm = diff_xx.subs([(x, xm), (y, ym)])
    print(A)
    print(fxxm)

xm = ...

if isinstance(xm, dict):
    xm = [[xm[x], xm[y]]]

for x_m, y_m in xm:

    diff_xx = ...
    diff_xy = ...
    diff_yy = ...
    D = ...
    A = D.subs([(x, x_m), (y, y_m)])

    if A > 0:
        diff_xx_solved = diff_xx.subs([(x, x_m), (y, y_m)])
        if diff_xx_solved > 0:
            # GTNN
            ...
        else:
            # GTLN
            ...
    elif A == 0:
        # KXD
        ...
    else:
        # YEN NGUA
        ...


# %%

def decipher(text, arr):
    x, y, z = arr
    key = abs(x**2 - y**2 - z)
    return "".join(chr(ord(c) ^ key) for c in text)


print(decipher("jgnnm", [1, 1, 2]))


# %%


def main(x, y):
    rv = list(zip(x, y))
    rv.sort(key=lambda x: x[0])
    return list(zip(*rv))


x = [2, 1, 4, 3]
y = [1, 3, 5, 8]
x, y = main(x, y)
print(x, y, sep="\n")

# %%


def main(N):
    N = str(N)
    if len(N) < 3:
        return False
    freqs = [0] * 10
    for c in N:
        freqs[int(c)] += 1

    odds = False
    ans = 0
    for freq in freqs:
        if freq % 2:
            odds = True
        ans += freq // 2
    return ans * 2 + odds


print(main(2131135))
print(main(11))
print(main(13113))
print(main(123121))
# %%
x = symbols("x")


def req8(f, eta, xi, tol):
    diff_x = diff(f, x, 1)
    for _ in range(10000):
        xi = xi - eta * diff_x.subs(x, xi).evalf()
        if abs(diff_x.subs(x, xi)) < tol:
            break
    return round(xi, 2)


print(req8(x**2 + 2*sin(x), 0.1, -5, 1e-3))
print(req8(x**2 + 2*x - 1, 0.1, -5, 1e-3))

# %%


def sol1(text):
    """
    Giải sử dụng list thuần, dễ convert sang các ngôn ngữ khác
    """
    freqs = [0] * 26
    offset = ord('a')
    for c in text:
        freqs[ord(c) - offset] += 1

    ans = []
    for i, val in enumerate(freqs):
        ans.append(chr(i + offset) * val)

    return ''.join(ans)


def sol(text):
    freqs = collections.Counter(text)
    return ''.join(c * freqs[c] for c in string.ascii_lowercase)


print(sol1("abdikjab"))
print(sol2("abdikjab"))


# %%


x, y, z, t = symbols("x, y, z, t")  # KHONG XOA


def req2(f, a, b, c):  # KHONG XOA

    d1 = diff(f, x)
    d2 = diff(f, y)
    d3 = diff(f, z)
    for d in d1, d2, d3:
        if not d.is_rational_function():
            return None
    d1s = d1.subs({x: float(a), y: float(b), z: float(c)})*(x-float(a))
    d2s = d2.subs({x: float(a), y: float(b), z: float(c)})*(y-float(b))
    d3s = d3.subs({x: float(a), y: float(b), z: float(c)})*(z-float(c))
    pttt = f.subs({x: float(a), y: float(b), z: float(c)}) + d1s + d2s + d3s
    return pttt


req2(abs(x), 1, 1, 1)
req2(x**2 + y ** 2 + z ** 2, 1, 1, 1)


# %%

def main(N):
    # Divisible by 9
    if sum(int(val) for val in str(N)) % 9:
        return -1
    freqs = collections.Counter(map(int, str(N)))

    # Divisible by 5
    ans = -1
    if freqs[0]:
        ans = "".join(str(i) * freqs[i] for i in reversed(range(10)))
    elif freqs[5]:
        freqs[5] -= 1
        ans = "".join(str(i) * freqs[i] for i in reversed(range(10)))
        ans += "5"
    return int(ans)


print(main(45))
print(main(153))
print(main(555554444))
print(main(12353215482))

# %%


def main(N, S, nums):
    """
    Recursive with memoizationi
    """
    visited = set()

    def dfs(i, target, sofar=[]):
        if (i, target) in visited:
            return None
        visited.add((i, target))
        if target == 0:
            return sofar
        if i == N - 1 or target < 0:
            return None
        return dfs(i + 1, target, sofar) or dfs(i + 1, target - nums[i], sofar + [nums[i]])

    return dfs(0, S)


def main2(N, S, nums):
    """
    DP
    """
    dp = [[[] for _ in range(N + 1)] for _ in range(S + 1)]
    for i, num in enumerate(nums, 1):
        for s in range(S + 1):
            if s >= num and (dp[s][i - 1] or s == S):
                dp[s - num][i] = dp[s][i - 1] + [num]
            dp[s][i] = dp[s][i - 1] + []
    for res in dp[0]:
        if res:
            return res


L = random.choices(range(10, 100), k=100)
print(main(10, 390, [200, 10, 20, 20, 50, 50, 50, 50, 100, 100]))
print(main2(10, 390, [200, 10, 20, 20, 50, 50, 50, 50, 100, 100]))


# %%

@timer
def main1(N):
    ans = []

    def dfs(total, vals=[]):
        if total > N:
            return None

        if total == N:
            ans.append(vals)
            return

        for num in range(1, N + 1):
            dfs(total + num, vals + [num])
    dfs(0)
    seen = set()
    for row in ans:
        seen.add(tuple(sorted(row)))
    return len(seen)


@timer
def main2(N):
    ans = []

    def dfs(total, vals=[]):
        if total > N:
            return None

        if total == N:
            ans.append(vals)
            return

        start = 1 if not vals else vals[-1]
        for num in range(start, N + 1):
            dfs(total + num, vals + [num])

    dfs(0)
    return len(ans)


"""
N = 4
dfs(0, 1): 5
    dfs(1, 1): 3
        dfs(2, 1): 2
            dfs(3, 1): 1
                dfs(4, 1): 1
                    <== 1 (1, 1, 1, 1)
                dfs(5, 2): 0
                dfs(6, 3): 0
                dfs(7, 4): 0
                <== 1
            dfs(4, 2): 1
                <== 1 (1, 1, 2)
            dfs(5, 3): 0
            ...
            <===
        dfs(3, 2): 0
            dfs(5, 2): 0
        dfs(4, 3): 1
            <== 1 (1, 3)
        ...
    dfs(2, 2): 1
        dfs(4, 2): 1
        <== 1 (2, 2)
    dfs(3, 3): 
        <== 0
    dfs(4, 4): 1
        <== 1
"""


@timer
def main3(N):
    # Time: O(N*N)
    @functools.cache
    def dfs(total, prev=1):
        if total > N:
            return 0
        if total == N:
            return 1
        rv = 0
        for num in range(prev, N + 1):
            rv += dfs(total + num, num)
        return rv
    return dfs(0)


@timer
def main4(N):
    # Time: O(N*N)
    dp = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        half = i // 2 + 1
        dp[i][half: i + 2] = [1] * half
        for j in reversed(range(i // 2 + 1)):
            dp[i][j] = dp[i - j][j] + dp[i][j + 1]
    return dp[N][1]


N = 15
print(main1(N))
print(main2(N))
print(main3(N))
print(main4(N))

# %%

idx = dict(zip(string.ascii_lowercase, range(26)))
print(idx)
s = "hello"
ans = 0
for c1, c2 in zip(s, s[1:]):
    ans += abs(idx[c1] - idx[c2])
print(ans)


# %%

s = "abcdef"
L = [1, 2, 3]
for c1, c2, c3 in zip(s, s[1:], s[2:]):
    print(c1, c2, c3)

# %%


def permutate(S):
    S = sorted(S)
    ans = []

    def dfs(i=0, group="", visited=set()):
        if len(group) == len(S):
            ans.append(group)
            return
        for i, c in enumerate(S):
            if i in visited:
                continue
            visited.add(i)
            dfs(i + 1, group + c, visited)
            visited.discard(i)
    dfs()
    return ans


print(permutate("4321"))
# print(list(itertools.permutations("aaa")))


# %%

def main(N):

    ans = []

    def dfs(target, sofar=[]):
        if target == 0:
            ans.append(sofar)
            return
        start = 0 if not sofar else sofar[-1]
        for n in range(start + 1, target + 1):
            dfs(target - n, sofar + [n])
    dfs(N)
    return ans


main(8)
# [[1, 2, 5], [1, 3, 4], [1, 7], [2, 6], [3, 5], [8]]


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


def main(R, C, K, mat):

    # Create list of (value, row, col)
    nums = []
    for r in range(R):
        for c in range(C):
            nums.append((mat[r][c], r, c))

    # Recursion to find combinations
    def dfs(step=0, total=0, rows=set(), cols=set()):
        if step == K:
            return total
        rv = 0
        for v, r, c in nums:
            if r in rows:
                continue
            if c in cols:
                continue
            rows.add(r)
            cols.add(c)
            rv = max(rv, dfs(step + 1, total + v, rows, cols))
            rows.discard(r)
            cols.discard(c)
        return rv

    return dfs()


main(3, 3, 2, [[1, 2, 3], [4, 5, 6], [7, 8, 9]])  # 14
main(5, 10, 3, [[70, 87, 78, 55, 67, 15, 91, 30, 49, 86],
                [61, 93, 83, 77, 63, 95, 34, 78, 52, 89],
                [19, 74, 24, 71, 46, 87, 91, 0, 45, 40],
                [11, 43, 85, 69, 60, 77, 8, 36, 48, 35],
                [56, 30, 11, 33, 75, 14, 10, 4, 22, 74]])  # 273
