# %%


"""
Question for interview:
Given the initial position of a knight on the board (5, 4)
and a sequence of legal moves A, B, C, D, E, F, G, H
return the final position of the knight after moving

Follow up question 1:
Update your program so that if there is any illegal move, 
the knight will not move at all (from its initial position)

Follow up question 2:
The input is "e4" instead of (5, 4)
Output should be "h8" instead of (8, 8)
"""


def main(pos="e4", moves="AGG"):
    delta = {"A": (-1, 2), "B": (-2, 1), "C": (-2, -1), "D": (-1, -2),
             "E": (1, -2), "F": (2, -1), "G": (2, 1), "H": (1, 2)}

    x, y = ord(pos[0]) - 65, int(pos[1])
    for move in moves:
        dx, dy = delta[move]
        x += dx
        y += dy
    return chr(x + 65) + str(y)


def q1(pos=(5, 4), moves="AGG"):
    """
    We want to see the candidate utilizes dictionary
    """
    delta = {"A": (-1, 2), "B": (-2, 1), "C": (-2, -1), "D": (-1, -2),
             "E": (1, -2), "F": (2, -1), "G": (2, 1), "H": (1, 2)}
    x, y = pos
    for move in moves:
        dx, dy = delta[move]
        x += dx
        y += dy
    return x, y


def q2(pos=(5, 4), moves="AGG"):
    """
    We want to see the candidate uses BFS or DFS
    """
    delta = {"A": (-1, 2), "B": (-2, 1), "C": (-2, -1), "D": (-1, -2),
             "E": (1, -2), "F": (2, -1), "G": (2, 1), "H": (1, 2)}
    x0, y0 = pos
    x1, y1 = q1(pos, moves)
    queue = [[x0, y0, 0]]
    for x, y, depth in queue:
        if x == x1 and y == y1:
            return depth == len(moves)
        for dx, dy in delta.values():
            queue.append([x + dx, y + dy, depth + 1])
    return False


# print(main("e4", "AGG"))  # h8
# print(main("e4", "AHCDEFACD"))  # a1
print(q2((5, 4), "AGG"))  # h8
print(q2((5, 4), "AHCDEFACD"))  # a1
