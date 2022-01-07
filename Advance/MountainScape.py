# %% Mountain Scape:
# https://py.checkio.org/en/mission/mountain-scape/


# new_pts = [((left := x - y) and 0) or (x, y) for x, y in reversed(pts) if x - y <= left]
def main(pts):
    pts.sort()

    # Filter out all triangles that are completely within triangles to its right
    new_pts = []
    left = float("inf")
    for x, y in reversed(pts):
        if x - y >= left:
            continue
        new_pts.append((x, y))
        left = x - y
    new_pts.reverse()

    # Filter out all triangles that are completely within triangles to its left
    pts = []
    right = float("-inf")
    for x, y in new_pts:
        if x + y <= right:
            continue
        pts.append((x, y))
        right = x + y

    area = pts[0][1] ** 2
    for (px, py), (x, y) in zip(pts, pts[1:]):
        ix = (px + py + x - y) / 2
        iy = max(py - (ix - px), 0)
        area += y * y - iy * iy
    return area


print(main([(1, 1), (4, 2), (7, 3)]))
print(main([(0, 2), (5, 3), (7, 5)]))
print(main([(1, 3), (5, 3), (5, 5), (8, 4)]))

print(main([[26, 8], [74, 16]]))
print(main([[10, 38], [98, 2], [36, 8], [14, 10], [92, 4],
      [59, 5], [78, 16], [68, 2], [48, 4], [37, 15]]))

print(main([(1, 3), (5, 3), (5, 5), (8, 4), (5, 1),
      (6, 2), (7, 3), (8, 4), (9, 3), (10, 2), (11, 1)]))
