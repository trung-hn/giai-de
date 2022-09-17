# %%

import math

# Time and Space: # O(N) N is type of bill. N = 9 in this problem


def take_bill_from_bank(bill, quantity, current):
    print("take", bill, quantity)
    banks[bill] -= quantity
    used[bill] += quantity
    return current - bill * quantity


def send_bill_to_bank(bill, quantity, current):
    print("give back", bill, quantity)
    banks[bill] += quantity
    used[bill] -= quantity
    return current + bill * quantity


def find_next_small_bill():
    for value, amt in banks.items():
        if amt > used[value]:
            return value


def move_money_until_enough(money, target):
    bills = reversed(list(banks.keys()))  # O(N)

    for bill in bills:  # O(N)
        if bill >= target:
            continue
        if money >= target:
            break
        avail = used[bill]
        delta = target - money
        quantity = min(avail, math.ceil(delta / bill))
        money = send_bill_to_bank(bill, quantity, money)
    money = take_bill_from_bank(target, 1, money)
    return money


money = 1000
banks = {1: 51, 2: 26, 5: 11, 10: 11, 20: 6, 50: 4, 100: 1, 200: 1, 500: 2}
used = {}

# cycle 1
for value, amt in banks.items():
    quantity = min(amt, money // value)
    money -= value * quantity
    used[value] = quantity

# cycle 2 -> inf
while money:
    bill = find_next_small_bill()  # O(N)
    money = move_money_until_enough(money, bill)  # O(N)


print(money)
print(used)
