# %%
import collections
import itertools
# %%


def prob1(A, B, C, D):
    if 0 in (A, B, C, D):
        return 0
    neg = sum(val < 0 for val in (A, B, C, D))
    return -1 if neg % 2 else 1


# %%

def prob3(N, K, nums):
    mx = max(itertools.chain(*nums))
    delta = [0] * (mx + 2)
    for start, end in nums:
        delta[start] += 1
        delta[end + 1] -= 1

    print(delta)

    for i in range(1, len(delta)):
        delta[i] += delta[i - 1]
    return delta.count(K)


print(prob3(3, 2, [[1, 5], [2, 8], [3, 7]]))

# %%


def prob4(N, M, K, graph, targets):
    neis = collections.defaultdict(list)
    for start, end in graph:
        neis[start].append(end)
        neis[end].append(start)
    max_dist_from_city = collections.defaultdict(int)

    def bfs(start):
        q = [(start, 0)]
        visited = set()
        for city, dist in q:
            visited.add(city)

            # Update distance
            max_dist_from_city[city] = max(max_dist_from_city[city], dist)

            # Visit neighbor cities
            for nei in neis[city]:
                if nei in visited:
                    continue
                q.append((nei, dist + 1))

    for city in targets:
        bfs(city)

    return [city for city, dist in max_dist_from_city.items() if dist <= K]


print(prob4(6, 2, 2, [[1, 2], [3, 2], [3, 5], [4, 2], [5, 6]], [1, 3]))
print(prob4(6, 3, 2, [[1, 2], [3, 2], [3, 5], [4, 1], [5, 6]], [1, 3, 5]))

# %%


class Prob5:
    def prob5(self, N, T, to_store1, to_store2, from_store1, from_store2):

        mem = {}
        visited = [0] * T
        self.ans = float("inf")

        def dfs(shipper=0, total=0, deliveries=0):
            if deliveries == T:
                self.ans = min(self.ans, total)
                return
            if total >= self.ans:
                return

            for cust in range(T):
                if visited[cust]:
                    continue
                for dist in options[(shipper, cust)]:
                    if dist:
                        visited[cust] = ~shipper
                        dfs(shipper + 1, total + dist, deliveries + 1)
                    else:
                        dfs(shipper + 1, total + dist, deliveries)
                    visited[cust] = 0

        options = collections.defaultdict(list)
        for n in range(N):
            for t in range(T):
                options[(n, t)].append(to_store1[n] + from_store1[t])
                options[(n, t)].append(to_store2[n] + from_store2[t])
                options[(n, t)].append(0)

        dfs()
        return self.ans


print(Prob5().prob5(3, 2, [1, 2, 2], [7, 5, 3], [3, 5], [2, 3]))

N = 5
T = 4
print(Prob5().prob5(N, T, [2]*N, [3]*N, [2]*T, [3]*T))
