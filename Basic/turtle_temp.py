# %%

import turtle

from numpy import square

turtle.clear()
for head in range(30, 360, 90):
    turtle.setheading(head)
    for _ in range(3):
        turtle.forward(100)
        turtle.right(120)
