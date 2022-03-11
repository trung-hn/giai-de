# %%

def main(S, K):
    stack = []
    for i, digit in enumerate(S):
        while stack and stack[-1] < digit \
                and len(stack) + len(S) - i > K:
            stack.pop()
        stack.append(digit)
    return "".join(stack)


print(main("82468", 3))
print(main("0123456", 1))
print(main("0123456", 2))
print(main("0123456", 3))
print(main("0123456", 4))
print(main("0123456", 5))
