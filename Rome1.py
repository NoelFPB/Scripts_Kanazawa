def solution(A, B):
    n = len(A)
    
    # Rome must be a city with no outgoing roads
    has_outgoing = [False] * (n + 1)
    for i in range(n):
        has_outgoing[A[i]] = True
    
    # find the unique city with no outgoing roads
    rome_candidate = -1
    for city in range(n + 1):
        if not has_outgoing[city]:
            if rome_candidate != -1:  # found multiple candidates
                return -1
            rome_candidate = city
    
    if rome_candidate == -1:  # no candidate found
        return -1
    
    # now verify this candidate is actually reachable from all cities
    # build adjacency list
    graph = [[] for _ in range(n + 1)]
    for i in range(n):
        graph[A[i]].append(B[i])
    
    # BFS from each city to see if it can reach rome_candidate
    from collections import deque
    
    for start in range(n + 1):
        if start == rome_candidate:
            continue
            
        visited = [False] * (n + 1)
        queue = deque([start]) # instead of a list that allows pop, this allows to have the pop on both sides on a O(n)
        visited[start] = True
        reached_rome = False
        
        while queue:
            current = queue.popleft() 
            if current == rome_candidate:
                reached_rome = True
                break
                
            for neighbor in graph[current]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
        
        if not reached_rome:
            return -1
    
    return rome_candidate





# less good 

def solution(A, B):
    n = len(A)
    
    # find cities that don't have any outgoing roads
    outgoing = set()
    for i in range(n):
        outgoing.add(A[i])
    
    possible = []
    for i in range(n + 1):
        if i not in outgoing:
            possible.append(i)
    
    if len(possible) != 1:
        return -1
    
    candidate = possible[0]
    
    # build graph
    adj = {}
    for i in range(n + 1):
        adj[i] = []
    for i in range(n):
        adj[A[i]].append(B[i])
    
    # check if everyone can reach the candidate
    def bfs(start, target):
        if start == target:
            return True
        q = [start]
        seen = set([start])
        
        while q:
            curr = q.pop(0)
            for next_city in adj[curr]:
                if next_city == target:
                    return True
                if next_city not in seen:
                    seen.add(next_city)
                    q.append(next_city)
        return False
    
    for city in range(n + 1):
        if city != candidate:
            if not bfs(city, candidate):
                return -1
    
    return candidate