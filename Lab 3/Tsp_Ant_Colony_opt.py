import random
import math

# -----------------------------
# PARAMETERS
# -----------------------------
NUM_CITIES = 10
NUM_ANTS = 20
ALPHA = 1.0        # pheromone importance
BETA = 5.0         # distance importance
RHO = 0.5          # evaporation rate
Q = 100            # pheromone deposit factor
ITERATIONS = 100

random.seed(1)

# -----------------------------
# CREATE TSP INSTANCE
# -----------------------------
def generate_cities(n, grid_size=100):
    """Randomly generate city coordinates."""
    return [(random.uniform(0, grid_size), random.uniform(0, grid_size)) for _ in range(n)]

def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def compute_distance_matrix(cities):
    n = len(cities)
    dist = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                dist[i][j] = distance(cities[i], cities[j])
    return dist

# -----------------------------
# ACO Core
# -----------------------------
def initialize_pheromones(n):
    """Initialize pheromone levels between cities."""
    return [[1.0 for _ in range(n)] for _ in range(n)]

def select_next_city(probabilities):
    """Roulette-wheel selection."""
    r = random.random()
    cumulative = 0.0
    for i, p in enumerate(probabilities):
        cumulative += p
        if r <= cumulative:
            return i
    return len(probabilities) - 1

def ant_tour(dist_matrix, pheromone):
    """Construct a complete tour for one ant."""
    n = len(dist_matrix)
    unvisited = list(range(n))
    start = random.choice(unvisited)
    tour = [start]
    unvisited.remove(start)

    while unvisited:
        current = tour[-1]
        probs = []
        denom = 0.0
        for j in unvisited:
            denom += (pheromone[current][j] ** ALPHA) * ((1.0 / dist_matrix[current][j]) ** BETA)
        for j in unvisited:
            num = (pheromone[current][j] ** ALPHA) * ((1.0 / dist_matrix[current][j]) ** BETA)
            probs.append(num / denom if denom > 0 else 0)
        next_city = unvisited[select_next_city(probs)]
        tour.append(next_city)
        unvisited.remove(next_city)
    return tour

def tour_length(tour, dist_matrix):
    """Calculate total length of the tour."""
    total = 0.0
    for i in range(len(tour) - 1):
        total += dist_matrix[tour[i]][tour[i+1]]
    total += dist_matrix[tour[-1]][tour[0]]  # return to start
    return total

def update_pheromones(pheromone, all_tours, all_lengths):
    """Evaporate and deposit pheromones based on ant tours."""
    n = len(pheromone)
    # Evaporation
    for i in range(n):
        for j in range(n):
            pheromone[i][j] *= (1 - RHO)
            if pheromone[i][j] < 1e-6:
                pheromone[i][j] = 1e-6

    # Deposit pheromones
    for tour, L in zip(all_tours, all_lengths):
        deposit = Q / L
        for i in range(len(tour) - 1):
            a, b = tour[i], tour[i+1]
            pheromone[a][b] += deposit
            pheromone[b][a] += deposit
        # return edge
        pheromone[tour[-1]][tour[0]] += deposit
        pheromone[tour[0]][tour[-1]] += deposit

# -----------------------------
# ACO MAIN LOOP
# -----------------------------
def aco_tsp(num_cities=NUM_CITIES):
    cities = generate_cities(num_cities)
    dist_matrix = compute_distance_matrix(cities)
    pheromone = initialize_pheromones(num_cities)

    best_tour = None
    best_length = float('inf')

    for it in range(ITERATIONS):
        all_tours = []
        all_lengths = []

        for ant in range(NUM_ANTS):
            tour = ant_tour(dist_matrix, pheromone)
            L = tour_length(tour, dist_matrix)
            all_tours.append(tour)
            all_lengths.append(L)

            if L < best_length:
                best_length = L
                best_tour = tour

        update_pheromones(pheromone, all_tours, all_lengths)

        if (it + 1) % 10 == 0 or it == 0:
            print(f"Iteration {it+1:3d} | Best Length: {best_length:.2f}")

    print("\n✅ Optimization Complete!")
    print(f"Best Tour Length: {best_length:.2f}")
    print(f"Best Tour: {best_tour}")
    return best_tour, best_length, cities

# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    best_tour, best_length, cities = aco_tsp()
