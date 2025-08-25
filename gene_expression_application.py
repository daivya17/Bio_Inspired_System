import random
import math

# ==============================================================================
# PARAMETERS
# ==============================================================================
POP_SIZE = 150
GENE_LEN = 12
# We are only optimizing one variable: the launch angle (theta).
NUM_GENES = 1
MUTATION_RATE = 0.05
CROSSOVER_RATE = 0.8
NUM_GENS = 20
# Search space for the launch angle (theta) in radians.
# We search from 0 to pi/2 (0 to 90 degrees).
SEARCH_MIN = 0.0
SEARCH_MAX = math.pi / 2.0
CHROMOSOME_LEN = GENE_LEN * NUM_GENES

# Constants for the projectile motion equation
VELOCITY = 100.0  # Initial velocity in m/s (held constant)
GRAVITY = 9.81  # Acceleration due to gravity in m/s^2

# ==============================================================================
# PROBLEM & FITNESS FUNCTION
# ==============================================================================
def projectile_range(angle):
    """
    Calculates the horizontal range of a projectile.
    
    Formula: R = (v^2 * sin(2*theta)) / g
    
    Args:
        angle (float): The launch angle in radians.
        
    Returns:
        float: The horizontal range.
    """
    return (VELOCITY**2 * math.sin(2 * angle)) / GRAVITY

def express_gene(gene):
    """
    Translates a binary gene to a real-valued number (the launch angle).
    """
    decimal_val = int("".join(map(str, gene)), 2)
    max_decimal = (2**GENE_LEN) - 1
    return SEARCH_MIN + (decimal_val / max_decimal) * (SEARCH_MAX - SEARCH_MIN)

def express_chromosome(chromosome):
    """
    Translates a full chromosome into a solution vector.
    Since we only have one gene, this returns a single value.
    """
    return express_gene(chromosome[0:GENE_LEN])

def evaluate_fitness(pop):
    """
    Evaluates fitness for the population.
    Since we are maximizing the range, fitness is simply the range value.
    We add 1 to avoid a zero or negative fitness score.
    """
    return [projectile_range(express_chromosome(c)) + 1 for c in pop]

# ==============================================================================
# GENETIC OPERATIONS
# ==============================================================================
def initialize_population(size, length):
    """Generates a random initial population of binary chromosomes."""
    return [[random.randint(0, 1) for _ in range(length)] for _ in range(size)]

def select_parents(pop, fitness_scores):
    """Selects two parents using roulette wheel selection."""
    total_fitness = sum(fitness_scores)
    if total_fitness <= 0:
        return random.choice(pop), random.choice(pop)

    weights = [f / total_fitness for f in fitness_scores]
    return random.choices(pop, weights=weights, k=2)

def crossover(p1, p2, rate):
    """Performs single-point crossover."""
    if random.random() < rate:
        pt = random.randint(1, CHROMOSOME_LEN - 1)
        return p1[:pt] + p2[pt:], p2[:pt] + p1[pt:]
    return p1, p2

def mutate(c, rate):
    """Mutates a chromosome by flipping bits."""
    return [1 - bit if random.random() < rate else bit for bit in c]

# ==============================================================================
# MAIN ALGORITHM LOOP
# ==============================================================================
def run_gea():
    """Main function to run the Gene Expression Algorithm."""
    pop = initialize_population(POP_SIZE, CHROMOSOME_LEN)
    best_solution, best_value = None, -1.0

    print("Optimizing for Maximum Projectile Range...")

    for gen in range(NUM_GENS):
        fitness_scores = evaluate_fitness(pop)
        
        current_best_index = fitness_scores.index(max(fitness_scores))
        current_best_angle_rad = express_chromosome(pop[current_best_index])
        current_best_range = projectile_range(current_best_angle_rad)

        if current_best_range > best_value:
            best_value = current_best_range
            best_solution = current_best_angle_rad

        new_pop = []
        # Elitism: Keep the best solution from the current population
        best_chromosome = pop[current_best_index]
        new_pop.append(best_chromosome)

        for _ in range((POP_SIZE - 1) // 2):
            p1, p2 = select_parents(pop, fitness_scores)
            o1, o2 = crossover(p1, p2, CROSSOVER_RATE)
            new_pop.append(mutate(o1, MUTATION_RATE))
            new_pop.append(mutate(o2, MUTATION_RATE))
        
        pop = new_pop
        
        best_solution_deg = math.degrees(best_solution)
        print(f"Gen {gen + 1}/{NUM_GENS} - Best Range: {best_value:.4f} m (at {best_solution_deg:.2f}°)")

    print("\n--- Optimization Complete ---")
    print(f"Final Best Angle: {math.degrees(best_solution):.2f}°")
    print(f"Final Best Range: {best_value:.4f} m")

if __name__ == "__main__":
    run_gea()
