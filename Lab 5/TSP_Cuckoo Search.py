import random
import math

# -------------------------------
# Helper functions
# -------------------------------

def distance(city1, city2):
    """Euclidean distance between two cities"""
    return math.sqrt((city1[0] - city2[0])**2 + (city1[1] - city2[1])**2)

def total_distance(path, cities):
    """Compute total distance of the path"""
    dist = 0
    for i in range(len(path)):
        city1 = cities[path[i]]
        city2 = cities[path[(i + 1) % len(path)]]
        dist += distance(city1, city2)
    return dist

def levy_flight(Lambda):
    """Generate step size using Lévy distribution"""
    sigma = (math.gamma(1 + Lambda) * math.sin(math.pi * Lambda / 2) /
             (math.gamma((1 + Lambda) / 2) * Lambda * 2**((Lambda - 1) / 2)))**(1 / Lambda)
    u = random.gauss(0, sigma)
    v = random.gauss(0, 1)
    step = u / abs(v)**(1 / Lambda)
    return step

def get_new_solution(path):
    """Generate a new random permutation by swapping two cities"""
    new_path = path[:]
    i, j = random.sample(range(len(new_path)), 2)
    new_path[i], new_path[j] = new_path[j], new_path[i]
    return new_path

# -------------------------------
# Cuckoo Search Algorithm
# -------------------------------

def cuckoo_search_tsp(cities, n_nests=15, pa=0.25, alpha=1, Lambda=1.5, max_iter=100):
    # Initialize nests (random permutations)
    nests = [random.sample(range(len(cities)), len(cities)) for _ in range(n_nests)]
    fitness = [total_distance(nest, cities) for nest in nests]

    best_nest = nests[fitness.index(min(fitness))]
    best_fitness = min(fitness)

    for iteration in range(max_iter):
        for i in range(n_nests):
            # Lévy flight (small random move)
            step_size = levy_flight(Lambda)
            new_nest = get_new_solution(nests[i])

            new_fitness = total_distance(new_nest, cities)

            # If better, replace
            if new_fitness < fitness[i]:
                nests[i] = new_nest
                fitness[i] = new_fitness

            # Update best
            if new_fitness < best_fitness:
                best_nest = new_nest[:]
                best_fitness = new_fitness

        # Abandon some nests with probability pa
        for i in range(n_nests):
            if random.random() < pa:
                nests[i] = random.sample(range(len(cities)), len(cities))
                fitness[i] = total_distance(nests[i], cities)

        print(f"Iteration {iteration+1}/{max_iter} - Best Distance: {best_fitness:.4f}")

    return best_nest, best_fitness

# -------------------------------
# Example Run
# -------------------------------

if __name__ == "__main__":
    # Example: 10 cities with random coordinates
    cities = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(10)]

    best_path, best_distance = cuckoo_search_tsp(cities, n_nests=20, max_iter=100)

    print("\n✅ Best Path Found:")
    print(best_path)
    print(f"Total Distance: {best_distance:.4f}")
