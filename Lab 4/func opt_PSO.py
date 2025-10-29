import random
import math

# Define the mathematical function to optimize (minimize)
def objective_function(x, y):
    return x**2 + y**2  # Sphere function

# PSO parameters
num_particles = 30
max_iterations = 100
w = 0.7      # inertia weight
c1 = 1.5     # cognitive (particle) weight
c2 = 1.5     # social (swarm) weight

# Initialize particles
particles = []
for i in range(num_particles):
    x = random.uniform(-10, 10)
    y = random.uniform(-10, 10)
    velocity = [random.uniform(-1, 1), random.uniform(-1, 1)]
    fitness = objective_function(x, y)
    particles.append({
        "position": [x, y],
        "velocity": velocity,
        "best_pos": [x, y],
        "best_fit": fitness
    })

# Initialize global best
global_best = min(particles, key=lambda p: p["best_fit"])
global_best_pos = global_best["best_pos"]
global_best_fit = global_best["best_fit"]

# Main PSO loop
for t in range(max_iterations):
    for particle in particles:
        # Update velocity
        r1, r2 = random.random(), random.random()
        for i in range(2):
            cognitive = c1 * r1 * (particle["best_pos"][i] - particle["position"][i])
            social = c2 * r2 * (global_best_pos[i] - particle["position"][i])
            particle["velocity"][i] = w * particle["velocity"][i] + cognitive + social

        # Update position
        particle["position"][0] += particle["velocity"][0]
        particle["position"][1] += particle["velocity"][1]

        # Evaluate new fitness
        fit = objective_function(particle["position"][0], particle["position"][1])

        # Update personal best
        if fit < particle["best_fit"]:
            particle["best_fit"] = fit
            particle["best_pos"] = particle["position"][:]

        # Update global best
        if fit < global_best_fit:
            global_best_fit = fit
            global_best_pos = particle["position"][:]

    print(f"Iteration {t+1}/{max_iterations} - Best Fitness: {global_best_fit:.6f}")

print("\n✅ Optimization complete!")
print(f"Best position found: {global_best_pos}")
print(f"Minimum value of function: {global_best_fit:.6f}")
