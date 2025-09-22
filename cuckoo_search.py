import numpy as np
import random

# ---- Distance function ----
def route_length(route, dist_matrix):
    length = 0
    for i in range(len(route)):
        length += dist_matrix[route[i-1]][route[i]]
    return length

# ---- Lévy flight operator (approximation for discrete TSP) ----
def levy_flight(route):
    new_route = route.copy()
    # simple swap of two cities (local search)
    i, j = random.sample(range(len(route)), 2)
    new_route[i], new_route[j] = new_route[j], new_route[i]
    return new_route

# ---- Cuckoo Search Algorithm ----
def cuckoo_search_tsp(dist_matrix, n=15, pa=0.25, max_iter=500):
    num_cities = len(dist_matrix)

    # Step 1: Initialize nests (random routes)
    nests = [random.sample(range(num_cities), num_cities) for _ in range(n)]
    fitness = [route_length(r, dist_matrix) for r in nests]

    best_route = nests[np.argmin(fitness)]
    best_cost = min(fitness)

    for _ in range(max_iter):
        # Step 2: Generate new solutions via Lévy flight
        cuckoo = levy_flight(best_route)
        cuckoo_cost = route_length(cuckoo, dist_matrix)

        # Step 3: Replace a random nest if cuckoo is better
        j = random.randint(0, n-1)
        if cuckoo_cost < fitness[j]:
            nests[j] = cuckoo
            fitness[j] = cuckoo_cost

        # Step 4: Abandon fraction Pa of worst nests
        abandon_count = int(pa * n)
        worst_indices = np.argsort(fitness)[-abandon_count:]
        for idx in worst_indices:
            nests[idx] = random.sample(range(num_cities), num_cities)
            fitness[idx] = route_length(nests[idx], dist_matrix)

        # Step 5: Update global best
        current_best_idx = np.argmin(fitness)
        if fitness[current_best_idx] < best_cost:
            best_route = nests[current_best_idx]
            best_cost = fitness[current_best_idx]

    return best_route, best_cost

# ---- Example Usage ----
if __name__ == "__main__":
    # Example 5 cities distance matrix (symmetric TSP)
    dist_matrix = [
        [0, 2, 9, 10, 7],
        [2, 0, 6, 4, 3],
        [9, 6, 0, 8, 5],
        [10, 4, 8, 0, 6],
        [7, 3, 5, 6, 0]
    ]

    best_route, best_cost = cuckoo_search_tsp(dist_matrix, n=20, pa=0.3, max_iter=1000)
    print("Best Route:", best_route)
    print("Best Cost:", best_cost)
