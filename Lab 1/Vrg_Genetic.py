"""
vrp_ga.py

Capacitated Vehicle Routing Problem (CVRP) solved using a Genetic Algorithm.

How it works (brief):
- Chromosome = permutation of customer indices (1..N). Depot is index 0.
- A greedy "split" transforms permutation -> list of routes by filling vehicles
  until capacity would be exceeded, then starts a new route.
- Fitness = total distance of all routes (lower is better).
- GA: initialization, tournament selection, Order Crossover (OX), swap mutation,
  replacement with elitism.

Run: python vrp_ga.py
"""

import math
import random
import copy
from typing import List, Tuple

random.seed(1)

# --------------------------
# Utility functions
# --------------------------

def euclidean(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def compute_distance_matrix(coords: List[Tuple[float, float]]) -> List[List[float]]:
    n = len(coords)
    dist = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            dist[i][j] = euclidean(coords[i], coords[j])
    return dist

# --------------------------
# Problem instance generator
# --------------------------

def generate_instance(num_customers=15, grid_size=100, max_demand=10, depot=(50,50)):
    """
    Generates random customers with demands and coordinates.
    Index 0 is the depot, customers are 1..num_customers
    """
    coords = [depot]
    demands = [0]
    for _ in range(num_customers):
        coords.append((random.uniform(0, grid_size), random.uniform(0, grid_size)))
        demands.append(random.randint(1, max_demand))
    return coords, demands

# --------------------------
# Chromosome -> Routes (split by capacity)
# --------------------------

def split_routes_from_perm(perm: List[int], demands: List[int], capacity: int) -> List[List[int]]:
    """
    Greedy split: take customer sequence in perm, keep adding to current route
    while capacity is not exceeded. Start new route when needed.
    """
    routes = []
    cur_route = []
    cur_load = 0
    for cust in perm:
        d = demands[cust]
        if cur_load + d <= capacity:
            cur_route.append(cust)
            cur_load += d
        else:
            # close route and start new
            if cur_route:
                routes.append(cur_route)
            cur_route = [cust]
            cur_load = d
    if cur_route:
        routes.append(cur_route)
    return routes

def route_cost(route: List[int], dist_matrix: List[List[float]]) -> float:
    if not route:
        return 0.0
    cost = 0.0
    prev = 0  # depot
    for cust in route:
        cost += dist_matrix[prev][cust]
        prev = cust
    cost += dist_matrix[prev][0]  # return to depot
    return cost

def total_cost(routes: List[List[int]], dist_matrix: List[List[float]]) -> float:
    return sum(route_cost(r, dist_matrix) for r in routes)

# --------------------------
# Genetic Algorithm components
# --------------------------

def init_population(pop_size: int, customer_ids: List[int]) -> List[List[int]]:
    pop = []
    for _ in range(pop_size):
        perm = customer_ids[:]
        random.shuffle(perm)
        pop.append(perm)
    return pop

def tournament_selection(pop: List[List[int]], fitnesses: List[float], k=3) -> List[int]:
    selected = random.sample(list(range(len(pop))), k)
    best = min(selected, key=lambda i: fitnesses[i])
    return copy.deepcopy(pop[best])

def order_crossover(parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
    """Order Crossover (OX) for permutations."""
    n = len(parent1)
    a, b = sorted(random.sample(range(n), 2))
    def ox(p1, p2):
        child = [None]*n
        # copy slice
        child[a:b+1] = p1[a:b+1]
        # fill remaining from p2 order
        pos = (b+1) % n
        p2_i = (b+1) % n
        while None in child:
            val = p2[p2_i]
            if val not in child:
                child[pos] = val
                pos = (pos+1) % n
            p2_i = (p2_i+1) % n
        return child
    return ox(parent1, parent2), ox(parent2, parent1)

def swap_mutation(perm: List[int], mut_rate=0.2) -> None:
    """In-place swap mutation (may perform multiple swaps based on mut_rate)."""
    n = len(perm)
    for i in range(n):
        if random.random() < mut_rate:
            j = random.randrange(n)
            perm[i], perm[j] = perm[j], perm[i]

# --------------------------
# GA main loop
# --------------------------

def evaluate_population(pop: List[List[int]], demands: List[int], capacity: int, dist_matrix: List[List[float]]):
    fitnesses = []
    for indiv in pop:
        routes = split_routes_from_perm(indiv, demands, capacity)
        fitnesses.append(total_cost(routes, dist_matrix))
    return fitnesses

def ga_vrp(
    coords: List[Tuple[float, float]],
    demands: List[int],
    capacity: int,
    pop_size=100,
    gens=300,
    crossover_rate=0.8,
    mutation_rate=0.15,
    elitism=2
):
    n = len(coords) - 1  # number of customers
    customer_ids = list(range(1, n+1))
    dist_matrix = compute_distance_matrix(coords)

    pop = init_population(pop_size, customer_ids)
    fitnesses = evaluate_population(pop, demands, capacity, dist_matrix)

    best_history = []
    for gen in range(gens):
        new_pop = []
        # Elitism: keep best individuals
        sorted_idx = sorted(range(len(pop)), key=lambda i: fitnesses[i])
        for i in range(elitism):
            new_pop.append(copy.deepcopy(pop[sorted_idx[i]]))

        while len(new_pop) < pop_size:
            # Selection
            p1 = tournament_selection(pop, fitnesses, k=3)
            p2 = tournament_selection(pop, fitnesses, k=3)
            # Crossover
            if random.random() < crossover_rate:
                c1, c2 = order_crossover(p1, p2)
            else:
                c1, c2 = p1, p2
            # Mutation
            swap_mutation(c1, mutation_rate)
            swap_mutation(c2, mutation_rate)
            new_pop.append(c1)
            if len(new_pop) < pop_size:
                new_pop.append(c2)

        pop = new_pop
        fitnesses = evaluate_population(pop, demands, capacity, dist_matrix)
        gen_best = min(fitnesses)
        best_history.append(gen_best)

        if (gen+1) % (max(1, gens//10)) == 0 or gen == 0:
            print(f"Gen {gen+1:4d} | best cost = {gen_best:.2f}")

    # final best
    best_idx = min(range(len(pop)), key=lambda i: fitnesses[i])
    best_perm = pop[best_idx]
    best_routes = split_routes_from_perm(best_perm, demands, capacity)
    best_cost = fitnesses[best_idx]
    return {
        'best_perm': best_perm,
        'best_routes': best_routes,
        'best_cost': best_cost,
        'dist_matrix': dist_matrix,
        'history': best_history
    }

# --------------------------
# Example / Run
# --------------------------

if __name__ == "__main__":
    # Example instance
    NUM_CUSTOMERS = 12
    VEHICLE_CAPACITY = 30
    coords, demands = generate_instance(num_customers=NUM_CUSTOMERS, grid_size=100, max_demand=12, depot=(50,50))

    print("Depot coords:", coords[0])
    for i in range(1, len(coords)):
        print(f"Customer {i:2d} at {coords[i]} demand={demands[i]}")

    result = ga_vrp(
        coords=coords,
        demands=demands,
        capacity=VEHICLE_CAPACITY,
        pop_size=120,
        gens=300,
        crossover_rate=0.9,
        mutation_rate=0.12,
        elitism=4
    )

    print("\n=== BEST SOLUTION ===")
    print(f"Total cost: {result['best_cost']:.2f}")
    print("Routes (customers listed, depot implicit at start/end):")
    for idx, r in enumerate(result['best_routes'], start=1):
        load = sum(demands[c] for c in r)
        rcost = route_cost(r, result['dist_matrix'])
        print(f" Vehicle {idx}: {r} | load={load} | route_cost={rcost:.2f}")
    print("\nBest permutation (customer visit order):", result['best_perm'])
