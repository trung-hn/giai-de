# %%
from datetime import datetime, timedelta
from math import floor


def main(n):
    day = n ** 0.5
    if day.is_integer():
        day -= 1
    day = floor(day)
    end = datetime(2022, 1, 1) + timedelta(days=day)
    return end.strftime("%d %m %Y"), (day + 1) ** 2


print(main(15))  # ('04 01 2022', 16)
print(main(961))  # ('31 01 2022', 961)
print(main(3601))  # ('02 03 2022', 3721)
