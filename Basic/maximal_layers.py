# %%
"""
Source: https://www.chegg.com/homework-help/questions-and-answers/consider-group-points-point-made-x-y-coordinate-point-group-considered-maximal-point-group-q28200462?fbclid=IwAR39R00TEM5if-hlWUwgNEb7LqFrLqwIYauJdTooel3Bgieb8McpVmRPB6A
"""
from collections import defaultdict


def main(points):
    point_no = len(points)
    sorted_points = defaultdict(list)
    # Step 1: Sort the Points
    for x, y in points:
        # Note that we don't store `y` because key is `y`
        sorted_points[y].append(x)

    for x_points in sorted_points.values():
        x_points.sort()

    y_levels = sorted(sorted_points.keys(), reverse=True)

    # Step 2: Assign points to layers
    maxmimal_layers = [[]]
    while point_no:
        right_limit = float("-inf")

        for y_level in y_levels:
            nxt_right_limit = right_limit

            for left_most_x in reversed(sorted_points[y_level]):
                if left_most_x < right_limit:
                    break  # Go to next y_level

                # Add point to this layer
                maxmimal_layers[-1].append((left_most_x, y_level))
                sorted_points[y_level].pop()
                point_no -= 1

                # Maintain right_limit
                nxt_right_limit = max(nxt_right_limit, left_most_x)

            # Replace right_limit
            right_limit = nxt_right_limit

        # Add [] for new layer
        maxmimal_layers.append([])
    return maxmimal_layers[:-1]


def formatted_print(maxmimal_layers):
    for i, layer in enumerate(maxmimal_layers, 1):
        print(f"Layer {i}:", end=" ")
        for x, y in layer:
            print(f"{x},{y}", end=" ")
        print()
    print()


formatted_print(main([(5, 5), (4, 9), (10, 2), (2, 3), (15, 7)]))
# Layer 1: 4,9 15,7
# Layer 2: 5,5 10,2
# Layer 3: 2,3
# Layer 4:

formatted_print(main([(5, 5), (10, 2), (2, 3), (15, 7), (2, 14), (1, 1),
                      (15, 2), (1, 7), (7, 7), (1, 4), (12, 10), (15, 15)]))
# Layer 1: 15,15 15,7 15,2
# Layer 2: 2,14 12,10
# Layer 3: 7,7 1,7 10,2
# Layer 4: 5,5
# Layer 5: 1,4 2,3
# Layer 6: 1,1


formatted_print(main([(6, 2), (13, 18), (9, 9), (20, 10), (19, 19), (12, 12),
                      (3, 3), (2, 15), (13, 13), (5, 12), (2, 14), (1, 20)]))
# Layer 1: 1,20 19,19 20,10
# Layer 2: 13,18 13,13
# Layer 3: 2,15 2,14 12,12 5,12
# Layer 4: 9,9
# Layer 5: 3,3 6,2
