import collections
# %%


def prob1(A, B, C, D):
    if 0 in (A, B, C, D):
        return 0
    neg = sum(val < 0 for val in (A, B, C, D))
    return -1 if neg % 2 else 1


# %%

def prob3(N, K, nums):
    mx = max(nums, key=lambda x: x[1])[1]
    delta = [0] * (mx + 2)
    for start, end in nums:
        delta[start] += 1
        delta[end + 1] -= 1

    print(delta)

    for i in range(1, len(delta)):
        delta[i] += delta[i - 1]
    return delta.count(K)


print(prob2(3, 2, [[1, 5], [2, 8], [3, 7]]))

# %%


def prob4(N, M, K, graph, targets):
    neis = collections.defaultdict(list)
    for start, end in graph:
        neis[start].append(end)
        neis[end].append(start)
    max_dist_to_city = collections.defaultdict(int)

    def bfs(start):
        q = [(start, 0)]
        visited = set()
        for city, dist in q:
            visited.add(city)

            # Update distance
            max_dist_to_city[city] = max(max_dist_to_city[city], dist)

            # Visit neighbor cities
            for nei in neis[city]:
                if nei in visited or nei == city:
                    continue
                q.append((nei, dist + 1))

    for city in targets:
        bfs(city)

    return [city for city, dist in max_dist_to_city.items() if dist <= K]


print(prob3(6, 2, 2, [[1, 2], [3, 2], [3, 5], [4, 2], [5, 6]], [1, 3]))
print(prob3(6, 3, 2, [[1, 2], [3, 2], [3, 5], [4, 1], [5, 6]], [1, 3, 5]))
