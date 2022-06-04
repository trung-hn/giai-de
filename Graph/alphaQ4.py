# %%
import heapq
import collections


def get_shortest_path_to_source(graph, node, sources):
    heap = [(0, node)]
    visited = set()
    while heap:
        cost_sofar, node = heapq.heappop(heap)

        #
        if node in visited:
            continue
        visited.add(node)

        if node in sources:
            return cost_sofar

        for neighbor, cost in graph[node]:
            heapq.heappush(heap, (cost_sofar + cost, neighbor))


# pre-process graph
nodes = list(range(1, 6 + 1))
edges = [(1, 2, 100), (1, 3, 100), (2, 3, 100), (1, 4, 500), (2, 5, 500), (3, 6, 500)]

graph = collections.defaultdict(list)

for a, b, c in edges:
    graph[a].append((b, c))
    graph[b].append((a, c))

# sources = [1, 2, 3]
sources = [1, 5, 6]

total = 0
for destination in nodes:
    rv = get_shortest_path_to_source(graph, destination, sources)
    total += rv
print(total)
