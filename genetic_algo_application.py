import random

def create_chromosome(length):
    return [random.choice([0, 1]) for _ in range(length)]

def calculate_fitness(chromosome):
    """Calculates the fitness (number of '1's) of a chromosome."""
    return sum(chromosome)

def select_parents(population, num_parents):
    """Selects parents based on their fitness (roulette wheel selection)."""
    total_fitness = sum(calculate_fitness(c) for c in population)
    selection_probs = [calculate_fitness(c) / total_fitness for c in population]
    
    parents = random.choices(population, weights=selection_probs, k=num_parents)
    return parents

def crossover(parent1, parent2):
    """Performs single-point crossover."""
    crossover_point = random.randint(1, len(parent1) - 1)
    
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    
    return child1, child2

def mutate(chromosome, mutation_rate):
    """Performs mutation by flipping bits with a given probability."""
    for i in range(len(chromosome)):
        if random.random() < mutation_rate:
            chromosome[i] = 1 - chromosome[i]  # Flip the bit
    return chromosome

def genetic_algorithm(chromosome_length, population_size, generations, mutation_rate):
    """Main genetic algorithm loop."""
    # 1. Initialization
    population = [create_chromosome(chromosome_length) for _ in range(population_size)]
    
    for generation in range(generations):
        print(f"--- Generation {generation+1} ---")
        
        # 2. Calculate fitness and select parents
        parents = select_parents(population, population_size)
        
        # 3. Create the new generation
        next_generation = []
        for i in range(0, population_size, 2):
            parent1 = parents[i]
            parent2 = parents[i+1]
            
            # Crossover
            child1, child2 = crossover(parent1, parent2)
            
            # Mutation
            next_generation.append(mutate(child1, mutation_rate))
            next_generation.append(mutate(child2, mutation_rate))
            
        population = next_generation
        
        # Find the best chromosome in the current generation
        best_chromosome = max(population, key=calculate_fitness)
        best_fitness = calculate_fitness(best_chromosome)
        
        print(f"Best chromosome: {best_chromosome}")
        print(f"Best fitness: {best_fitness}")
        
        # 4. Termination condition
        if best_fitness == chromosome_length:
            print(f"Optimal solution found!")
            return best_chromosome
            
    # Return the best chromosome found after all generations
    return max(population, key=calculate_fitness)

# Parameters
CHROMOSOME_LENGTH = 16
POPULATION_SIZE = 100
GENERATIONS = 200
MUTATION_RATE = 0.01

# Run the algorithm
best_solution = genetic_algorithm(CHROMOSOME_LENGTH, POPULATION_SIZE, GENERATIONS, MUTATION_RATE)
print("\n--- Final Results ---")
print("Final best solution:", best_solution)
print("Fitness:", calculate_fitness(best_solution))
