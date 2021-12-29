# %% BRACKET

# Time: O(N)

def main(N, arr):
    cnt = 0
    stack = []
    rv = ""
    for val in arr:
        rv += "("
        stack.append(val)
        while stack and stack[-1] == 0:
            stack.pop()
            cnt += 1
            if stack:
                stack[-1] -= cnt * 2
            rv += ")"
        cnt = 0
    return rv


print(main(7, [4, 2, 0, 2, 0, 0, 0]))
print(main(10, [8, 2, 0, 0, 0, 4, 0, 0, 0, 0]))
