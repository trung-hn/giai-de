# %% Challenge
from functools import reduce


def main1(arr):
    return [num for val in arr for num in main(val)] if isinstance(arr, list) else [arr]


def main2(prev, curr):
    return prev + reduce(main2, curr, []) if isinstance(curr, list) else [curr]


def main3(arr):
    rv = []
    stack = [arr]
    while stack:
        if stack[-1] == []:
            stack.pop()
        elif isinstance(stack[-1], list):
            stack.append(stack[-1].pop())
        else:
            rv.append(stack.pop())
    return list(reversed(rv))


arr = [
    1,
    [2, [3, [4, [5]]]],
    [[[[6]]]],
    [
        [7],
        [8],
        [
            [
                9,
            ],
            [[10]],
        ],
    ],
]

print(main1(arr))
# [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

print(reduce(main2, arr, []))
# [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

print(main3(arr))
# [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
