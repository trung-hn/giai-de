# %% Number of numbers that are to the left and smaller

import collections


def main(nums):
    def merge(nums):
        if len(nums) == 1:
            return nums
        pivot = len(nums) // 2
        left = merge(nums[:pivot])
        right = merge(nums[pivot:])
        for i in reversed(range(len(nums))):
            if len(right) == 0 or left and left[-1] < right[-1]:
                counter[left[-1]] += len(right)
                nums[i] = left.pop()
            else:
                nums[i] = right.pop()
        return nums

    counter = [0] * (max(nums) + 1)
    merge(list(reversed(nums)))
    return [counter[val] for val in nums]


main([4, 3, 6, 2, 1, 5])  # [0, 1, 0, 3, 4, 1]
