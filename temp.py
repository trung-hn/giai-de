# %%
from typing import NamedTuple
from tkinter.tix import Tree
import turtle
import bisect
from audioop import reverse
import bisect
import datetime as dt
from itertools import takewhile
import enum
import itertools
import os
from turtle import st
from typing import NamedTuple, overload
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


def custom_range(arr):
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


print(custom_range([7, 2, 28, 5, 8, 4, 20, 10, 12, 15]))
print(custom_range([7, 3, 2, 11, 13]))

# %% Bai 4


def custom_range(N, arr):
    ans = [0] * N
    for val in arr:
        ans[val] = 1
    return sum(ans)


print(custom_range(11, [1, 2, 3, 4, 5, 1, 2, 1, 2, 7, 5]))


# %% Bai 1

def dfs(fib, N, ptr=0, sofar=[]):
    if N == 0:
        return sofar
    for i, val in enumerate(fib[ptr:], ptr):
        if rv := dfs(fib, N - val, i + 1, sofar + [val]):
            return rv


def custom_range(N):
    fib = [1, 2]
    while fib[-1] < N:
        fib.append(fib[-1] + fib[-2])

    fib.reverse()
    return dfs(fib, N)


print(custom_range(16))
print(custom_range(33))

# %% Bài 3


def custom_range(arr):

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


print(custom_range([-7, 4, -8, 9, 25, 17]))
print(custom_range([6, 3, 8, 16, -5, 8, -6, 4, -9, -4, -1, -5]))

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


def custom_range(N, M, nums):
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


def custom_range(N):
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


print(*custom_range(6), sep="\n")

# %%


def sort_num(num):
    freqs = [0] * 10
    while num:
        freqs[num % 10] += 1
        num //= 10
    return (str(i) * freqs[i] for i in range(1, 10))


def custom_range(N):
    a = 1
    for _ in range(N - 1):
        a = int("".join(sort_num(a * 2)))
    return a


print(custom_range(7))


# %%

def custom_range(N):
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


print(custom_range(45))
print(custom_range(153))
print(custom_range(555554444))
print(custom_range(12353215482))

# %%


def custom_range(N, S, nums):
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


def sol(N, S, nums):
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
print(custom_range(10, 390, [200, 10, 20, 20, 50, 50, 50, 50, 100, 100]))
print(sol(10, 390, [200, 10, 20, 20, 50, 50, 50, 50, 100, 100]))


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
def sol(N):
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
print(sol(N))
print(main3(N))
print(main4(N))

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

def custom_range(N):

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


custom_range(8)
# [[1, 2, 5], [1, 3, 4], [1, 7], [2, 6], [3, 5], [8]]

# %%


@timer
def custom_range(R, C, K, mat):
    seen_rows = [0] * R
    seen_cols = [0] * C

    # Recursion to find combinations
    visited = {}

    def dfs(step=0, total=0):
        key = hash((step, tuple(seen_rows), tuple(seen_cols)))
        if key in visited:
            return visited[key]

        if step == K:
            visited[key] = total
            return total
        rv = 0
        for r in range(R):
            if seen_rows[r]:
                continue
            seen_rows[r] = 1
            for c in range(C):
                if seen_cols[c]:
                    continue
                seen_cols[c] = 1
                rv = max(rv, dfs(step + 1, total + mat[r][c]))
                seen_cols[c] = 0
            seen_rows[r] = 0
        visited[key] = rv
        return rv

    return dfs()


custom_range(3, 3, 2, [[1, 2, 3], [4, 5, 6], [7, 8, 9]])  # 14
custom_range(5, 10, 3, [[70, 87, 78, 55, 67, 15, 91, 30, 49, 86],
                        [61, 93, 83, 77, 63, 95, 34, 78, 52, 89],
                        [19, 74, 24, 71, 46, 87, 91, 0, 45, 40],
                        [11, 43, 85, 69, 60, 77, 8, 36, 48, 35],
                        [56, 30, 11, 33, 75, 14, 10, 4, 22, 74]])  # 273

N = 9
custom_range(N, N, N, [[1] * N] * N)

# %%


def check(step, arr):
    """
    Step is  how many "1" moves
    arr is the current state
    """
    original = pos = arr.index(1)
    for i in range(step):
        pos = (pos + 1) % 6
        if i == step - 1:
            arr[pos] = 1
            arr[original] = 0
        if arr[pos] == 2:
            break
    return arr


print(check(1, [2, 0, 0, 0, 1, 0]))  # [2, 0, 0, 0, 0, 1]
print(check(2, [2, 0, 0, 0, 1, 0]))  # [1, 0, 0, 0, 0, 1]
print(check(3, [2, 0, 0, 0, 1, 0]))  # [2, 0, 0, 0, 1, 0]

# %%

file = input().lower()
print("YES" if file.endswith(".py") else "NO")

# %%
a = [0, 1, 1, 2, 4, 1, 5, 5, 8]
b = []
for i in a[::-1]:
    c = i + 1
    j = 0
    while j <= c:
        if j in b:
            c += 1
        j += 1
    b.append(c)
print(b[::-1])
# [1, 5, 3, 4, 8, 2, 7, 6, 9]

# %%


@timer
def custom_range(A, B):
    dp = [set() for _ in range(B + 1)]
    target = ans = 0
    for num in range(2, B + 1):
        # DP
        for val in range(2, int(num ** 0.5) + 1):
            if num % val == 0:
                dp[num] = dp[val] | dp[num // val]
                break
        # Prime number
        if not dp[num]:
            dp[num] = {num}
        # Find answer
        if num > A and len(dp[num]) >= target:
            target = len(dp[num])
            ans = (num, target)
    return ans


print(custom_range(2, 13))  # (12, 2)
print(custom_range(1000, 2000))  # (1995, 4)
print(custom_range(1, 100000))  # (99330, 6)

# %% invalid non-printable character U+200B
# print​(​"Hello"​)​

# %%


def custom_range(father=26, son=5):
    diff = father - 2 * son
    return father, son, diff, (father + diff) == 2 * (son + diff)


for _ in range(20):
    son = random.randint(1, 100)
    father = random.randint(max(son * 2, son + 25), 300)
    print(custom_range(father, son))

# %%

# %%


def custom_range(N, K):
    while 1:
        # Frog finds good place
        if not K % N:
            return N

        # Frog jump left until it cannot
        K %= N

        # Frog jump right once and become type M (K is M)
        N, K = K, K + N


print(custom_range(2, 10))
print(custom_range(2, 11))

# Đáp án đây nhưng em phải ngẫm một chút thì may ra mới hiểu được. Mà đề hơi khó hiểu nên đây chỉ là theo cách hiểu đề của anh thôi. Đó là:

# Ếch nhảy qua trái để khi không thể thì thôi. Khi đó, ếch sẽ nhảy qua phải 1 lần (bước nhảy N), rồi trở thành con ếch với bước nhảy mới (M, cũng là vị trí trước khi nhảy qua phải).

# Sau đó cứ thế lặp lại


# %%
for N in (18, 63, 73, 91, 438, 122690412):
    for val in range(1, int(N ** 0.5) + 1):
        if N % val == 0:
            print(N, val)

 # %%


def is_prime(N):
    if N == 2:
        return True
    return all(N % i for i in range(2, int(N ** 0.5) + 1))


@timer
def sol(arr):
    visited = {}

    def dfs(cnt, prev, used):
        # Memoization
        key = hash(tuple(used))
        if key in visited:
            return visited[key]

        # Base case
        if cnt == len(arr):
            visited[key] = 1
            return 1

        # Recursive case
        ans = 0
        for i, num in enumerate(arr):
            # If used or 2 numbers sum to prime, skip
            if used[i] or prev and is_prime(prev + num):
                continue
            used[i] = 1
            ans += dfs(cnt + 1, num, used)
            used[i] = 0

        visited[key] = ans
        return ans
    used = [0] * len(arr)
    return dfs(0, None, used)


print(sol([2] * 16))  # 2432902008176640000 finished in 14.287252 secs

# %%


def custom_range(R, C, mat):

    def neis(r, c):
        return [(x, y) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                if 0 <= (x := r + dx) < R and 0 <= (y := c + dy) < C
                and not mat[x][y]]

    def dfs(r=0, c=0):
        mat[r][c] = 1
        if not neis(r, c):
            return dead_ends.append((r + 1, c + 1))

        for x, y in neis(r, c):
            if not mat[x][y]:
                dfs(x, y)

    ones = sum(itertools.chain(*mat))
    dead_ends = []
    dfs()
    return sum(itertools.chain(*mat)) - ones, *sorted(dead_ends)


print(*custom_range(4, 7, [[0, 1, 1, 1, 0, 1, 0], [1, 0, 0, 0, 1, 1, 0],
                           [0, 1, 1, 0, 1, 1, 0], [1, 1, 1, 1, 0, 1, 0]]), sep="\n")
# 8
# (1, 5)
# (3, 1)
# (4, 5)

# %%
n = int(input())
print((6, 8, 4, 2)[n % 4] if n else 1)


# %%

def custom_range(N='1000'):
    num = ["khong", "mot", "hai", "ba", "bon",
           "nam", "sau", "bay", "tam", "chin"]
    deg = ["ngan", "tram", "muoi", '']
    ans = []
    for i, c in enumerate(N):
        if i == 2 and c == 0:
            ans.append("le")
        ans.append(num[int(c)] + " " + deg[i])

    return ' '.join(ans)


print(custom_range())

# %%


def custom_range():
    N = 100000
    counter = [0] * 13
    for _ in range(N):
        counter[sum(random.choices(range(1, 7), k=2))] += 1

    for i, cnt in enumerate(counter):
        if i > 1:
            print(f"{i} {cnt / N * 100:.2f}")


custom_range()
# %%

T = ["Không", "Một", "Hai", "Ba", "Bốn", "Năm", "Sáu", "Bảy", "Tám", "Chín"]


def r2_digits(t, c):
    out = "{} Mươi {}".format(T[t], T[c])

    if c == 0:
        out = "{} Mươi".format(T[t])
        if t == 1:
            out = "Mười"
    elif t == 1:
        out = "Mười {}".format(T[c])
        if c == 5:
            out = "Mười Lăm"

    return out


def r3_digits(t, c, d):
    out = "{} Trăm {}".format(T[t], r2_digits(c, d))

    if c == 0:
        out = "{} Trăm Lẻ {}".format(T[t], T[d])
        if d == 0:
            out = "{} Trăm".format(T[t])
    elif d == 1:
        out = "{} Trăm {} Mươi Mốt".format(T[t], T[c])
        if c == 1:
            out = "{} Trăm Mười {}".format(T[t], T[d])
    elif d == 0:
        out = "{} Trăm {} Mươi".format(T[t], T[c])
        if c == 1:
            out = "{} Trăm Mười".format(T[t])

    return out


def r4_digits(*ds):
    n, *r = ds
    out = "{} Ngàn {}".format(T[n], r3_digits(*r))

    if all(x == 0 for x in r):
        out = "{} Ngàn".format(T[n])

    return out


for n in range(1100, 1111):
    if len(str(n)) == 1:
        print(T[n])
    else:
        print(eval(f"r{len(str(n))}_digits(*map(int, str(n)))"))


# %%
def solution(arr):
    return sum(v1 >= v2 for v1, v2 in zip(arr, arr[1:])) < 2


print(solution([1, 3, 2, 1]))
print(solution([1, 3, 2]))


# %%

def solution(arr):
    c1 = sum(v1 >= v2 for v1, v2 in zip(arr, arr[1:])) < 2
    c2 = sum(v1 >= v2 for v1, v2 in zip(arr, arr[2:])) < 2
    return c1 and c2


# %%


def sol1(s="12341238950791823476123"):
    cnt = collections.Counter(s)
    for i in range(10):
        print(f"{i}:{cnt[str(i)]}")


def sol2(s="12341238950791823476123"):
    cnt = [0] * 10
    for val in s:
        cnt[int(val)] += 1
    print(*enumerate(cnt), sep="\n")


sol1()
sol2()
# %%


def custom_range(k):
    nums = []
    i = 0
    while len(nums) < k:
        ln = len(str(i))
        if sum(int(d) for d in str(i)) % ln == 0:
            nums.append(i)
        i += 1
    print(nums)
    return nums[-1]


print(custom_range(4000))

# %%
dx = 0
dx = 1

# up
[(0, 1), (1, 0), (0, -1), (-1, 0)]

# %%


def custom_range(M=3, N=2):

    @functools.cache
    def dfs(i, gifts, prev=float("inf")):
        if i == N:
            return gifts == 0

        rv = 0
        for amt in reversed(range(min(prev, gifts) + 1)):
            rv += dfs(i + 1, gifts - amt, amt)
        return rv
    return dfs(0, M)


custom_range(5, 4)

# %%


def custom_range(M=3, N=2):
    dp = [[0] * (M + 1) for _ in range(N, j + 1)]
    dp[1] = [1] * (M + 1)
    for n in range(2, N + 1):
        for m in range(1, M + 1):
            if m == 1:
                dp[n][m] = 1
                continue
            for k in range(math.floor(m/n + 1), m + 1):
                dp[n][m] += dp[n - 1][k]

    print(*dp, sep="\n")
    return dp[-1][-1]

# %%


def custom_range(arr):
    seen = [0] * (10**3 + 1)
    for n in arr:
        seen[n] = True

    nums = [i for i, val in enumerate(seen) if val]
    return len(nums), nums


custom_range([1, 2, 6, 4, 2, 3, 1, 3])


# %%


def custom_range(arr):
    arr.sort()
    ans = arr[0][1] - arr[0][0]
    for p1, p2 in zip(arr, arr[1:]):
        if p1[1] >= p2[0]:
            p2[0] = p1[0]
            p2[1] = max(p2[1], p1[1])
        ans = max(ans, p2[1] - p2[0])
    return ans


custom_range([[-1, 2], [1, 5], [6, 7], [-3, -2]])  # 6
custom_range([[-1, 10], [1, 3], [10, 12]])


# %%

def custom_range(nums):
    freqs = collections.Counter(nums)
    ans = []
    for num in freqs:
        if freqs[num] == 1 and freqs[num + 1] == freqs[num - 1] == 0:
            ans.append(num)
    return ans


custom_range([10, 6, 5, 8])

# %%


def custom_range(R, C, mat):
    def dfs(val, r, c):
        mat[r][c] = size = 0
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                x = r + dr
                y = c + dc
                if (dr or dc) and 0 <= x < R and 0 <= y < C and mat[x][y] == val:
                    size += dfs(val, x, y)
        return size + 1

    freqs = collections.Counter()
    ans = 0
    for r in range(R):
        for c in range(C):
            val = mat[r][c]
            if val and (size := dfs(val, r, c)) > 1:
                ans = max(ans, size)
                freqs[val] += 1
    print(sum(freqs.values()), ans, freqs, sep="\n")


custom_range(5, 6, [[1, 2, 1, 4, 5, 4], [1, 3, 1, 4, 5, 4],
                    [3, 1, 2, 3, 3, 4], [2, 2, 4, 5, 3, 3], [2, 2, 3, 2, 1, 1]])
# 8
# 5
# Counter({1: 2, 4: 2, 3: 2, 5: 1, 2: 1})


# %%
def main(N, T):
    delta = 2 * math.pi / T
    for i in range(N + 1):
        space = round((math.sin(delta * i) + 1) * T)
        print(" " * space + str(i))


main(20, 5)

# %%


class Student:
    def __init__(self, name, age, Id):
        self.name = name
        self.age = age
        self.Id = Id

    # Getter
    def get_name(self):
        return self.name

    def get_age(self):
        return self.age

    def get_Id(self):
        return self.Id

    # Setter
    def set_name(self, name):
        self.name = name

    def set_age(self, age):
        self.age = age

    def set_Id(self, Id):
        self.Id = Id

    def __eq__(self, other):
        return vars(self) == vars(other)

    def check(self, name, age, Id):
        return self == Student(name, age, Id)


student = Student(1, 2, 3)
student.check(1, 2, 3)


# %%

def main(nums):
    ans = [0] * 3
    nums.append(-1)
    stack = [0]
    for right, num in enumerate(nums):
        while stack and nums[stack[-1]] > num:
            height = nums[stack.pop()]
            left = stack[-1]
            total = height * (right - left - 1)
            if total > ans[0]:
                ans = [total, left + 2, right]
        stack.append(right)
    return ans


main([3, 4, 3, 1])
main([1, 2, 1, 3])

# %%


def get_forest_level(forest):
    init = power = 0
    for monster in forest:
        if power <= monster:
            init += (monster - power + 1)
            power = monster + 1
        power += 1
    return init


def main(N, K, forests):
    levels = []
    for forest in forests:
        level = get_forest_level(forest)
        levels.append(level)

    init = power = 0
    for level in sorted(levels):
        if power < level:
            init += (level - power)
            power = level
        power += K
    return init


print(main(2, 5, [[3, 3, 3, 3, 3], [0, 1, 2, 3, 4]]))
print(main(2, 2, [[10, 15, 8], [12, 11, 7]]))
print(main(1, 5, [[1, 2, 3, 4, 5, 6]]))
print(main(1, 5, [[6, 5, 4, 3, 2, 1]]))
print(main(3, 3, [[8, 9, 10, 9], [1, 2, 3], [3, 1, 2]]))

"""
Lý thuyết:

Trước hết, mình tính sức mạnh bắt đầu cần thiết để tiêu diệt hết quái trong mỗi khu rừng:
Ví dụ: 
[8, 9, 10, 9] cần bắt đầu với 9 sức mạnh.
[1, 2, 3] cần bắt đầu với 2 sức mạnh.
[3, 1, 2] cần bắt đầu với 4 sức mạnh.

Tính xong thì ta sẽ có list: [9, 2, 4]. A gọi tạm đây là độ khó của từng khu rừng.

Vì mình biết là sau khi diệt hết quái ở mỗi khu rừng thì ta sẽ mạnh lên K điểm, nên cách tối ưu nhất là vô từng khu rừng từ dễ đến khó một.

Nên e chỉ cần sort độ khó của từng khu rừng và check sức mạnh hiện tại so với độ khó thôi. Nếu sức mạnh >= độ khó thì ta sẽ giải quyết được khu rừng này và tăng sức mạnh lên K điểm. Còn không được thì ta sẽ cần tăng điểm ban đầu lên sao cho vừa đủ với độ khó. Cứ lặp như thế đến hết

Ở ví dụ này thì sẽ là:
ban đầu init = power = 0
vào rừng độ khó 2: Vì power < 2 nên power += 2 (bằng 2) và init += 2 (bằng 2) 
sau khi diệt hết quái, power = 2 + K = 5

vào rừng độ khó 4: Vì power > 4, k cần làm gì đặc biệt
sau khi diệt hết quái, power = 5 + K = 8

vào rừng độ khó 9: Vì power < 9, nên power += 1 (bằng 9) và init += 1 (bằng 3) 
sau khi diệt hết quái, power = 9 + K = 12

Trả về init = 3
"""

# %%


def main(N, edges, states):
    neis = collections.defaultdict(list)
    for a, b in edges:
        neis[a - 1].append(b - 1)
        neis[b - 1].append(a - 1)

    for start in range(N):

        q = [start]
        visited = [0] * N
        for node in q:
            # for nei in
            visited[]
            nxt = set()
            for nei in neis[node]:

            q = nxt


main(5, [[1, 2], [1, 3], [1, 4], [1, 5]], [1, 1, 0, 0, 0])

# %%


def main(N):
    seen = [0] * 11
    ans = []

    def dfs(i=0, seq=0):
        if i == N:
            ans.append(seq)
            return
        for n in range(1, N + 1):
            if seen[n]:
                continue
            seen[n] = 1
            dfs(i + 1, seq * 10 + n)
            seen[n] = 0
    dfs()
    return ans


print(main(2))  # [12, 21]
print(main(3))  # [123, 132, 213, 231, 312, 321]

# %%


def time_left_msg(time_curr, time_start):    # Tính khoảng thời gian chuẩn bị vào học
    time_1 = dt.timedelta(hours=int(time_curr[:2]), minutes=int(time_curr[2:]))
    time_2 = dt.timedelta(
        hours=int(time_start[:2]), minutes=int(time_start[2:]))
    minute = (time_2 - time_1).seconds // 60
    hh, mm = minute // 60, minute % 60
    if hh:
        return f"con {hh} gio {mm} phut"
    return f"con {mm} phut"


def in_class(schedules, day, time, end, i):
    msg = time_left_msg(time, end)
    if i != len(schedules) - 1:
        return "Dang hoc " + schedules[day][i] + f", {msg}" + ", tiet sau la " + schedules[day][i+1]
    return "Dang hoc " + schedules[day][i] + f", {msg} la tan hoc"


def not_in_class(subject, time, start):
    msg = time_left_msg(time, start)
    if i == 4:
        return f"Dang nghi trua, {msg}, tiet sau la " + subject
    return f"Dang ra choi, {msg}, tiet sau la " + subject


def main(day, time):

    periods = [["0730", "0815"], ["0825", "0910"], ["0920", "1005"], ["1015", "1100"],
               ["1330", "1415"], ["1425", "1510"], ["1520", "1605"], ["1615", "1700"]]     # Khoảng thời gian các tiết học

    starts = ["0730", "0825", "0920", "1015", "1330", "1425", "1520", "1615"]
    ends = ["0815", "0910", "1005", "1100", "1415", "1510", "1605", "1700"]

    schedules = [["Van", "Van", "Anh", "Anh", "Toan", "Toan", "Ly", "Ly"],   # TKB các buổi học
                 ["CD", "CN", "TC Ly", "TC Ly", "PW", "PW", "Toan", "Toan"],
                 ["Van", "Van", "Anh", "Anh", "Su", "Sinh", "Tin", "Tin"],
                 ["Van", "Van", "TC Hoa", "TC Sinh", "PW", "PW", "TD", "TD"],
                 ["Toan", "Toan", "Dia", "Dia", "Anh", "Anh", "Hoa", "Hoa"]
                 ]

    if day >= 5:
        return "Dang la ngay nghi"

    if time < periods[0][0]:
        msg = time_left_msg(time, periods[0][0])
        return f"Chua vao hoc, {msg} nua, tiet dau la " + schedules[day][0]

    elif time >= periods[-1][-1]:
        return "Da het gio hoc"

    for i, (start, end) in enumerate(periods):
        if time < start:
            return not_in_class(schedules[day][i], time, start)
        elif time < end:
            return in_class(schedules, day, time, end, i)


weekday = dt.date.today().weekday()  # 0 - 6 : Monday - Sunday

# chẳng hạn thời gian thực tại là 12:58, time_now là 1258
time_now = dt.datetime.now().strftime("%H%M")

print(main(0, "1040"))

# %%


def prob3(nums):
    nums.sort(reverse=True)
    return sum(max(0, num - i) for i, num in enumerate(nums))


print(prob3([4, 4, 4, 4]))  # -> 10
print(prob3([7, 5, 3, 1]))  # -> 12


def prob4(arr):
    arr.sort(key=lambda x: (-x[1], -x[0]))
    ans = 0
    allow = 1
    for i, (money, bag) in enumerate(arr):
        if i == allow:
            break
        ans += money
        allow += bag
    return ans


print(prob4([[1, 0], [2, 0], [0, 2]]))  # -> 3
print(prob4([[0, 1], [0, 2], [1, 0], [1, 0], [1, 0], [4, 0]]))  # -> 5

# %%

# ws = turtle.Screen()
t = turtle.Turtle()
for i in range(6):
    t.forward(100)  # Assuming the side of a hexagon is 100 units
    t.right(60)  # Turning the turtle by 60 degree
# %%


class Temp1:
    def main(self, arr, N, M):
        self.max_total = self.ans = -1
        visited = set()

        def dfs(start=0, step=0, total=0):
            if total > M:
                return
            if step == N:
                if self.max_total < total:
                    self.max_total = total
                    self.ans = tuple(visited)
                return

            for i in range(start, len(arr)):
                if i in visited:
                    continue
                visited.add(i)
                dfs(i + 1, step + 1, total + arr[i])
                visited.discard(i)
        dfs()
        return self.ans


class Temp:
    def main(self, arr, N, M):
        self.max_total = self.ans = -1
        visited = set()

        def dfs(i=0, total=0):
            if i >= len(arr):
                return
            if total > M:
                return
            if len(visited) == N:
                if self.max_total < total:
                    self.max_total = total
                    self.ans = tuple(visited)
                return

            dfs(i + 1, total)
            visited.add(i)
            dfs(i + 1, total + arr[i])
            visited.discard(i)
        dfs()
        return self.ans


Temp().main([1, 4, 3, 5, 11]*10, 7, 13)


# %%
def main(items, queries):
    items.append([0, 0])
    items.sort(key=lambda x: (x[0], -x[1]))
    for i in range(1, len(items)):
        items[i][1] = max(items[i][1], items[i - 1][1])

    ans = []
    for q in queries:
        idx = bisect.bisect(items, [q, float("inf")])
        ans.append(items[idx - 1][1])
    return ans


main([[1, 2], [3, 2], [2, 4], [5, 6], [3, 5]], [1, 2, 3, 4, 5, 6])

# %%


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

# %%


def search(target, base, lo=1, hi=100_000_000):
    while lo < hi:
        mid = (lo + hi) // 2
        prod = base * mid
        if prod == target:
            return mid
        if prod < target:
            lo = mid + 1
        else:
            hi = mid
    return lo


@timer
def cal1(N):
    end = n = curr = 1
    arr = [1]
    while curr < N:
        n += 1
        idx = search(end + 1, n)
        start = idx * (n)
        end = start + (n - 1) * n
        arr += list(range(start, end + 1, n))
        curr += n
    # print(arr)
    return arr[N - 1]


print(cal1(1000))
# %%


def search(target, base, lo=1, hi=10**12):
    while lo < hi:
        mid = (lo + hi) // 2
        prod = base * mid
        if prod == target:
            return mid
        if prod < target:
            lo = mid + 1
        else:
            hi = mid
    return lo


@timer
def cal(N):
    end = n = cnt = 1
    while cnt < N:
        n += 1
        end = (end // n + n) * n
        cnt += n
    return end - (cnt - N) * n


cal(1000)  # 942808708248388310, in 0.213027 secs

# %%


def sort_odd_even(arr):
    odds = [i + 1 for i in range(len(arr)) if arr[i] % 2 == i % 2 == 1]
    evens = [i + 1 for i in range(len(arr)) if arr[i] % 2 == i % 2 == 0]
    return list(zip(evens, odds))


sort_odd_even([1, 11, 12, 40, 6, 5, 25, 18])  # [(3, 2), (5, 6)]
# %%


def two_prod(square, N):
    for n in range(2, int(square ** 0.5) + 1):
        div = square // n
        if square % n == 0 and div != n and div <= N:
            print(n, div)


def main(N, K):

    squares = [i * i for i in range(2, N)]
    if K == 1:
        return sum(square <= N for square in squares)
    if K == 2:
        for square in squares:
            two_prod(square, N)
    if K == 3:
        ...


print(main(16, 2))

# %%


def main(N, nums):
    div = min(num // N for num in nums)
    nums = [num - div * N for num in nums]
    for i, num in enumerate(nums * 2):
        if i >= num:
            return i % N + 1


print(main(4, [2, 3, 2, 0]))  # 3
print(main(4, [12, 13, 11, 10]))  # 4
print(main(4, [12, 13, 11, 12]))  # 1
print(main(4, [13, 13, 11, 12]))  # 2

# %%


def main(N, nums):
    min_div = N
    for i, num in enumerate(nums * 3):
        if i < N:
            min_div = min(min_div, nums[i] // N)
        if i >= num - min_div * N + N:
            return i % N + 1


print(main(4, [2, 3, 2, 0]))     # 3
print(main(4, [12, 13, 11, 10]))  # 4
print(main(4, [12, 13, 11, 12]))  # 1
print(main(4, [13, 13, 11, 12]))  # 2

# %%


def main(black, white):
    pts = [(p, -1) for p in black] + [(p, 1) for p in white]
    pts.sort()
    k = 0
    curr = None
    for _, type in pts:
        if curr is None:
            curr = type
            continue
        if curr != type:
            k += 1
            curr = None
    return k


main([-6, 3, -3, 5, -4], [-7, 2, -1, -8, 1])  # 3

# %%


def main(arr, target):
    arr = list(map(int, arr))
    trackers = [False] * len(arr)

    @functools.lru_cache(None)
    def dp(i=0, curr=0, total=0):
        if i == len(arr):
            if total == target:
                return trackers.copy()
            return

        trackers[i] = True
        used = dp(i + 1, 0, total + curr)
        trackers[i] = False
        skip = dp(i + 1, curr * 10 + arr[i], total)
        return used or skip

    signs = dp()
    ans = ""
    curr = 0
    for digit, sign in zip(arr, signs):
        if sign:
            ans += f"{curr}+"
            curr = 0
        curr = curr * 10 + digit
    return ans[:-1]


main("3207011864", 32)  # '32+0+7+0+1+1+8+6'

# %%


def main(S):
    ans = []
    for k, g in itertools.groupby(S):
        length = len(list(g))
        ans.append(f"{length}{k}" if length > 1 else str(k))
    return ''.join(ans)


main("aabbbcaAA")


# %%


class Thing(NamedTuple):
    name: str
    value: int


v1 = Thing("a", 1)
v2 = Thing("b", 2)

# %%


def main(nums):
    freq = collections.Counter(nums)
    sorted_freq = sorted(freq, key=freq.get, reverse=True)
    return [val * freq[val] for val in sorted_freq]


def main(nums):
    first_seen = {}
    freqs = [0] * 10
    for i, num in enumerate(nums):
        if num not in first_seen:
            first_seen[num] = i
        freqs[num] += 1

    sorted_freq = sorted(enumerate(freqs), key=lambda x: -x[1])
    return [num * freq for num, freq in sorted_freq if freq]


print(main([5, 6, 8, 6, 7, 4, 5, 3, 4, 5, 6, 7, 8, 9, 6]))
# [24, 15, 8, 14, 16, 3, 9]

# %%


def main(n, a, b):
    minutes_per_day = a - b
    return math.ceil((n - a) / minutes_per_day) + 1


main(16, 7, 2)

# %%


