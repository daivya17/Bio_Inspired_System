import numpy as np
import random

class ACO_TSP:
    def __init__(self, dist_matrix, num_ants=10, alpha=1, beta=5, rho=0.5, Q=100, iterations=100):
        self.dist = dist_matrix
        self.n = len(dist_matrix)
        self.num_ants = num_ants
        self.alpha = alpha      # pheromone importance
        self.beta = beta        # heuristic importance
        self.rho = rho          # evaporation rate
        self.Q = Q              # pheromone deposit factor
        self.iterations = iterations

        # Initialize pheromone levels (small constant)
        self.pheromone = np.ones((self.n, self.n)) / self.n

        self.best_length = float('inf')
        self.best_path = None

    def run(self):
        for it in range(self.iterations):
            all_paths = []
            all_lengths = []

            # Each ant builds a tour
            for _ in range(self.num_ants):
                path = self.construct_solution()
                length = self.path_length(path)

                all_paths.append(path)
                all_lengths.append(length)

                # Update global best
                if length < self.best_length:
                    self.best_length = length
                    self.best_path = path

            # Update pheromone matrix
            self.update_pheromone(all_paths, all_lengths)

            print(f"Iteration {it+1}/{self.iterations} | Best length: {self.best_length}")

        return self.best_path, self.best_length

    def construct_solution(self):
        path = []
        visited = set()
        current = random.randint(0, self.n - 1)
        path.append(current)
        visited.add(current)

        while len(visited) < self.n:
            probabilities = []
            for j in range(self.n):
                if j not in visited:
                    tau = self.pheromone[current][j] ** self.alpha
                    eta = (1 / self.dist[current][j]) ** self.beta
                    probabilities.append(tau * eta)
                else:
                    probabilities.append(0)

            probabilities = np.array(probabilities)
            probabilities /= probabilities.sum()  # normalize probabilities

            next_city = np.random.choice(range(self.n), p=probabilities)
            path.append(next_city)
            visited.add(next_city)
            current = next_city

        return path

    def path_length(self, path):
        length = 0
        for i in range(len(path)):
            length += self.dist[path[i]][path[(i + 1) % self.n]]  # return to start
        return length

    def update_pheromone(self, paths, lengths):
        # Evaporation
        self.pheromone *= (1 - self.rho)

        # Deposit pheromone
        for path, length in zip(paths, lengths):
            for i in range(len(path)):
                a, b = path[i], path[(i + 1) % self.n]
                self.pheromone[a][b] += self.Q / length
                self.pheromone[b][a] += self.Q / length  # symmetric TSP

# Example usage
if __name__ == "__main__":
    dist_matrix = np.array([
        [0, 2, 9, 10],
        [1, 0, 6, 4],
        [15, 7, 0, 8],
        [6, 3, 12, 0]
    ])

    aco = ACO_TSP(dist_matrix, num_ants=5, alpha=1, beta=5, rho=0.5, Q=100, iterations=30)
    best_path, best_length = aco.run()

    print("\nBest Path:", best_path)
    print("Best Length:", best_length)
