import numpy as np

# Knapsack fitness function
def fitness(solution, values, weights, capacity):
    total_value = np.sum(solution * values)
    total_weight = np.sum(solution * weights)
    if total_weight > capacity:
        return 0  # infeasible
    return total_value

# Repair function (greedy remove until feasible)
def repair(solution, values, weights, capacity):
    while np.sum(solution * weights) > capacity:
        idx = np.where(solution == 1)[0]
        if len(idx) == 0:
            break
        # remove least value/weight ratio item
        worst = min(idx, key=lambda i: values[i]/weights[i])
        solution[worst] = 0
    return solution

# Grey Wolf Optimizer for Knapsack
def GWO_knapsack(values, weights, capacity, n_wolves=10, max_iter=5):
    n_items = len(values)

    # Initialize wolves (random binary solutions)
    wolves = np.random.randint(0, 2, (n_wolves, n_items))
    for i in range(n_wolves):
        wolves[i] = repair(wolves[i], values, weights, capacity)

    # Evaluate fitness
    fitness_vals = [fitness(w, values, weights, capacity) for w in wolves]

    # Identify alpha, beta, delta
    sorted_idx = np.argsort(fitness_vals)[::-1]
    alpha = wolves[sorted_idx[0]].copy()
    beta  = wolves[sorted_idx[1]].copy()
    delta = wolves[sorted_idx[2]].copy()
    alpha_score = fitness_vals[sorted_idx[0]]

    # Main loop
    for t in range(max_iter):
        a = 2 - 2*(t/max_iter)  # linearly decreases from 2 to 0

        for i in range(n_wolves):
            X = wolves[i].copy()
            X_new = np.zeros(n_items)

            for d in range(n_items):
                r1, r2 = np.random.rand(), np.random.rand()
                A1 = 2*a*r1 - a
                C1 = 2*r2
                D_alpha = abs(C1*alpha[d] - X[d])
                X1 = alpha[d] - A1*D_alpha

                r1, r2 = np.random.rand(), np.random.rand()
                A2 = 2*a*r1 - a
                C2 = 2*r2
                D_beta = abs(C2*beta[d] - X[d])
                X2 = beta[d] - A2*D_beta

                r1, r2 = np.random.rand(), np.random.rand()
                A3 = 2*a*r1 - a
                C3 = 2*r2
                D_delta = abs(C3*delta[d] - X[d])
                X3 = delta[d] - A3*D_delta

                # Average position
                X_new[d] = (X1 + X2 + X3) / 3

                # Convert to binary (sigmoid + threshold)
                prob = 1 / (1 + np.exp(-X_new[d]))
                X_new[d] = 1 if np.random.rand() < prob else 0

            # Repair and evaluate
            X_new = repair(X_new.astype(int), values, weights, capacity)
            new_score = fitness(X_new, values, weights, capacity)

            # Replace if better
            if new_score > fitness_vals[i]:
                wolves[i] = X_new
                fitness_vals[i] = new_score

        # Update Alpha, Beta, Delta
        sorted_idx = np.argsort(fitness_vals)[::-1]
        alpha = wolves[sorted_idx[0]].copy()
        beta  = wolves[sorted_idx[1]].copy()
        delta = wolves[sorted_idx[2]].copy()
        alpha_score = fitness_vals[sorted_idx[0]]

        # 🔹 Print best solution of this iteration
        print(f"Iteration {t+1}: Best Solution = {alpha}, Best Value = {alpha_score}")

    return alpha, alpha_score


# Example usage
if __name__ == "__main__":
    values = np.array([15, 10, 9, 5, 8])
    weights = np.array([1, 5, 3, 4, 2])
    capacity = 8

    best_solution, best_value = GWO_knapsack(values, weights, capacity, n_wolves=5, max_iter=8)
    print("\nFinal Best Solution:", best_solution)
    print("Final Best Total Value:", best_value)
