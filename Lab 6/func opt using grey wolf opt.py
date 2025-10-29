import random
import math

# -----------------------------
# Objective Function (to minimize)
# -----------------------------
def objective_function(position):
    # Sphere function: sum of squares
    return sum(x**2 for x in position)

# -----------------------------
# Grey Wolf Optimizer
# -----------------------------
def grey_wolf_optimizer(num_wolves=20, dim=2, max_iter=100, lower_bound=-10, upper_bound=10):
    # Initialize the population of wolves randomly
    wolves = [[random.uniform(lower_bound, upper_bound) for _ in range(dim)] for _ in range(num_wolves)]
    fitness = [objective_function(w) for w in wolves]

    # Initialize alpha, beta, delta (best 3 wolves)
    alpha, beta, delta = [0]*dim, [0]*dim, [0]*dim
    alpha_score, beta_score, delta_score = float("inf"), float("inf"), float("inf")

    # Main loop
    for t in range(max_iter):
        for i in range(num_wolves):
            score = fitness[i]

            # Update alpha, beta, delta
            if score < alpha_score:
                delta_score, delta = beta_score, beta[:]
                beta_score, beta = alpha_score, alpha[:]
                alpha_score, alpha = score, wolves[i][:]
            elif score < beta_score:
                delta_score, delta = beta_score, beta[:]
                beta_score, beta = score, wolves[i][:]
            elif score < delta_score:
                delta_score, delta = score, wolves[i][:]

        # Linearly decreasing a from 2 to 0
        a = 2 - t * (2 / max_iter)

        # Update each wolf's position
        for i in range(num_wolves):
            new_position = []
            for j in range(dim):
                r1, r2 = random.random(), random.random()
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = abs(C1 * alpha[j] - wolves[i][j])
                X1 = alpha[j] - A1 * D_alpha

                r1, r2 = random.random(), random.random()
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = abs(C2 * beta[j] - wolves[i][j])
                X2 = beta[j] - A2 * D_beta

                r1, r2 = random.random(), random.random()
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = abs(C3 * delta[j] - wolves[i][j])
                X3 = delta[j] - A3 * D_delta

                new_x = (X1 + X2 + X3) / 3
                # Keep within bounds
                new_x = max(lower_bound, min(upper_bound, new_x))
                new_position.append(new_x)

            wolves[i] = new_position
            fitness[i] = objective_function(new_position)

        print(f"Iteration {t+1}/{max_iter} - Best fitness: {alpha_score:.6f}")

    return alpha, alpha_score

# -----------------------------
# Example Run
# -----------------------------
if __name__ == "__main__":
    best_position, best_score = grey_wolf_optimizer(num_wolves=20, dim=5, max_iter=100)
    print("\n✅ Optimization complete!")
    print(f"Best position found: {best_position}")
    print(f"Minimum value of function: {best_score:.6f}")
