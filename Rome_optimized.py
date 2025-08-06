from collections import deque, defaultdict

def solution(A, B):
    n = len(A)
    total_cities = n + 1

    # 1. Find the city with no outgoing edges
    has_outgoing = [False] * total_cities
    for i in range(n):
        has_outgoing[A[i]] = True

    rome_candidate = -1
    for city in range(total_cities):
        if not has_outgoing[city]:
            if rome_candidate != -1:
                return -1  # multiple candidates
            rome_candidate = city

    if rome_candidate == -1:
        return -1

    # 2. Build reverse graph: reverse_graph[to] = list of froms
    reverse_graph = defaultdict(list)
    for i in range(n):
        reverse_graph[B[i]].append(A[i])

    # 3. Do BFS in reverse from rome_candidate
    visited = [False] * total_cities
    queue = deque([rome_candidate])
    visited[rome_candidate] = True
    count = 1  # count of how many cities can reach Rome

    while queue:
        current = queue.popleft()
        for prev_city in reverse_graph[current]:
            if not visited[prev_city]:
                visited[prev_city] = True
                queue.append(prev_city)
                count += 1

    if count == total_cities:
        return rome_candidate
    else:
        return -1
