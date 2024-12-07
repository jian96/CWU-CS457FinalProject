import random
import csv
import numpy as np
from datetime import datetime
from scipy import stats
import matplotlib.pyplot as plt
import os
import cProfile
import re
from sudoku import Sudoku
import math
import time

RED = "\033[31m"
GREEN = "\033[32m"
RESET = "\033[0m"

# def print_sudoku(state):
#     border = "------+-------+------"
#     rows = [state[i:i+9] for i in range(0,81,9)]
#     for i,row in enumerate(rows):
#         if i % 3 == 0:
#             print(border)
#         three = [row[i:i+3] for i in range(0,9,3)]
#         print(" | ".join(
#             " ".join(str(x or "_") for x in one)
#             for one in three
#         ))
#     print(border)

def generate_fixed_mask(board):
    """
    Generates a fixed mask for a Sudoku board based on non-zero values.
    
    Args:
        board (np.ndarray): A 1D NumPy array of length 81 representing a Sudoku board.
        
    Returns:
        np.ndarray: A 1D NumPy boolean array of length 81, where `True` indicates a fixed value.
    """
    return board != 0  # Returns a boolean mask where True indicates non-zero values

def coord(row, col):
    return row*9+col

def block_indices(block_num):
    """return linear array indices corresp to the sq block, row major, 0-indexed.
    block:
       0 1 2     (0,0) (0,3) (0,6)
       3 4 5 --> (3,0) (3,3) (3,6)
       6 7 8     (6,0) (6,3) (6,6)
    """
    firstrow = (block_num // 3) * 3
    firstcol = (block_num % 3) * 3
    indices = [coord(firstrow+i, firstcol+j) for i in range(3) for j in range(3)]
    return indices

def generate_individual(problem):
    solution = problem.copy()
    for block in range(9):
        indices = block_indices(block)
        block = problem[indices]
        zeros = [i for i in indices if problem[i] == 0]
        to_fill = [i for i in range(1, 10) if i not in block]
        random.shuffle(to_fill)
        for index, value in zip(zeros, to_fill):
            solution[index] = value
    return solution

def calculate_fitness(individual):
    """calculate the number of violations: assume all rows are OK"""
    column_score = lambda n: len(set(individual[coord(i, n)] for i in range(9)))
    row_score = lambda n: len(set(individual[coord(n, i)] for i in range(9)))
    fitness_score = sum(column_score(n)+row_score(n) for n in range(9))
    return fitness_score

# single-point crossover function
# parents can fail to mate, function will keep trying random crossover points
# to see if the children are valid.
# if it's valid returns children and success bool
# if it's not valid then returns last batch of kids and failure bool

def crossover(board1, board2):
    """
    Performs crossover on two Sudoku boards using a random crossover point.
    Generates two child boards by swapping grids.
    
    Args:
        board1 (np.ndarray): A 1D NumPy array of length 81 representing the first Sudoku board.
        board2 (np.ndarray): A 1D NumPy array of length 81 representing the second Sudoku board.
        
    Returns:
        np.ndarray, np.ndarray: Two new 1D NumPy arrays representing the resulting Sudoku boards.
    """
    # Reshape boards into 9x9 grids
    board1 = board1.reshape(9, 9)
    board2 = board2.reshape(9, 9)
    
    # Choose a random crossover point (1-8 inclusive)
    crossover_point = np.random.randint(1, 9)
    
    # Initialize the child boards as copies of the parents
    child1 = board1.copy()
    child2 = board2.copy()
    
    # Loop through the 9 grids
    for grid in range(9):
        # Determine the starting row and column for the current grid
        start_row = (grid // 3) * 3
        start_col = (grid % 3) * 3
        
        # If the grid index is greater than or equal to the crossover point, swap grids
        if grid >= crossover_point:
            # Swap grids between the children
            child1[start_row:start_row + 3, start_col:start_col + 3], \
            child2[start_row:start_row + 3, start_col:start_col + 3] = \
            child2[start_row:start_row + 3, start_col:start_col + 3], \
            child1[start_row:start_row + 3, start_col:start_col + 3]
    
    # Flatten the boards back to 1D
    return child1.flatten(), child2.flatten()

# swap mutation function
# swaps 2 random elements within the matrix, can randomly swap 1-n elements with max_mutation

def mutate(board, fixed_mask, mutation_rate):
    # Decide if mutation should occur based on the mutation rate
    if np.random.rand() > mutation_rate:
        return board  # No mutation; return the board unchanged

    # Reshape the board and fixed mask into 9x9 grids
    board = board.reshape(9, 9)
    fixed_mask = fixed_mask.reshape(9, 9)
    
    # Select a random 3x3 grid
    grid_row = np.random.randint(0, 3)  # Grid row index (0, 1, 2)
    grid_col = np.random.randint(0, 3)  # Grid column index (0, 1, 2)
    
    # Extract the starting row and column for the 3x3 grid
    start_row = grid_row * 3
    start_col = grid_col * 3
    
    # Get the sub-grid and its fixed mask
    sub_grid = board[start_row:start_row + 3, start_col:start_col + 3].flatten()
    sub_grid_mask = fixed_mask[start_row:start_row + 3, start_col:start_col + 3].flatten()
    
    # Identify mutable indices (non-fixed values) in the sub-grid
    mutable_indices = np.where(~sub_grid_mask)[0]
    
    # If there are fewer than 2 mutable elements, skip mutation
    if len(mutable_indices) < 2:
        return board.flatten()

    # Randomly select two different mutable indices to swap
    idx1, idx2 = np.random.choice(mutable_indices, size=2, replace=False)
    
    # Perform the swap
    sub_grid[idx1], sub_grid[idx2] = sub_grid[idx2], sub_grid[idx1]
    
    # Put the mutated sub-grid back into the board
    board[start_row:start_row + 3, start_col:start_col + 3] = sub_grid.reshape(3, 3)
    
    # Return the mutated board as a flattened array
    return board.flatten()

# def mean_absolute_deviation(numbers):
#     mean = sum(numbers) / len(numbers)
#     return sum(abs(x - mean) for x in numbers) / len(numbers)


# def dynamic_mutation_rates(mean_history, lookback_period, mutation_rate):
#     lookback_window = mean_history[-lookback_period:]
#     stuck = False
#     if len(lookback_window) == 1:
#         return mutation_rate, stuck
#     mad = mean_absolute_deviation(lookback_window)
#     if mad <= 0.06:
#         mutation_rate += 0.5
#         stuck = True
#         print("Stagnation detected, incrementing mutation rates to " + str(mutation_rate))
#     else:
#         mutation_rate = 0.1
#         stuck = False
#         # print("Resetting mutation rates to" + str(mutation_rate))
#     return mutation_rate, stuck


def genetic_algorithm(problem, n, population_size, generations, mutation_rate, max_mutations, cull_percent, restart_gen):
    
    random.seed()
    np.random.seed()
    restart_num = 0
    success = False
    fixed_mask = generate_fixed_mask(problem)
    # generate initial population, ensuring uniqueness (not really needed)
    
    best_so_far = 0
    gens_without_improve = 0
    max_fitness = 0
    
    population_set = set()
    while not len(population_set) == population_size:
        population_set.add(tuple(generate_individual(problem)))

    population = np.array(list(population_set))
    # population = np.array([np.reshape(lst, (n, n)) for lst in population])
    fitness_scores = [calculate_fitness(individual)for individual in population]

    fittest_individual_log = []
    fittest_score_log = []
    fitness_mean_log = []

    for generation in range(generations):
        #print(f"Generation {generation}, restarted {restart_num} times")
        # print("gen: " + str(generation))
        # Crossover and mutation
        new_population_set = set()
        while not len(new_population_set) >= population_size:
            crossover_success = False

            # makes sure crossover is successful, because some pairs cannot create valid children
            while (not crossover_success):
                # roulette selection - selects parents based on fitness score
                parent1 = random.choices(
                    population, weights=fitness_scores, k=1)[0]
                parent2 = random.choices(
                    population, weights=fitness_scores, k=1)[0]

                # makes sure parent1 and parent2 aren't just the same individual
                while np.array_equal(parent1, parent2):
                    parent2 = random.choices(
                        population, weights=fitness_scores, k=1)[0]

                # crossover, children can be invalid magic squares, in that case disgard and choose another parent pair
                child1, child2 = crossover(parent1, parent2)
                crossover_success = True

                # Evaluate the fitness of the parents and children
                parent1_fitness = calculate_fitness(parent1)  # Replace with your fitness evaluation function
                parent2_fitness = calculate_fitness(parent2)  # Replace with your fitness evaluation function
                child1_fitness = calculate_fitness(child1)    # Replace with your fitness evaluation function
                child2_fitness = calculate_fitness(child2)    # Replace with your fitness evaluation function
                print("Starting genetic algorithm")
                # Find the individual with the highest fitness
                all_candidates = [
                    (parent1, parent1_fitness),
                    (parent2, parent2_fitness),
                    (child1, child1_fitness),
                    (child2, child2_fitness)
                ]
                best_individual, best_individual_fitness = max(all_candidates, key=lambda x: x[1])
            best_individual_mutated = mutate(best_individual, fixed_mask, mutation_rate)
            best_individual_mutated_fitness = calculate_fitness(best_individual_mutated)
            #child2 = mutate(child2, fixed_mask, mutation_rate)
            #new_population_set.add(tuple(child1))
            if (best_individual_mutated_fitness > best_individual_fitness):
                new_population_set.add(tuple(best_individual_mutated))
            else:
                new_population_set.add(tuple(best_individual))

        # update old pop with new pop and calculate new pop's fitness scores
        population = np.array(list(new_population_set))
        #population = np.array([np.reshape(lst, (n, n)) for lst in population])
        fitness_scores = [calculate_fitness(
            individual) for individual in population]

        # make the ind + score [individual, fitness] list
        population_with_scores = population
        for i in range(len(population_with_scores)):
            population_with_scores = [(individual, fitness_scores[i]) for i, individual in enumerate(population)]

        # sort the ind + score by score, remove score and reshape to nxn
        # need this for culling, culls x% bottom up starting from lowest fitness
        population_with_scores = sorted(
            population_with_scores, key=lambda x: x[-1], reverse=True)
        population_with_scores_reshaped = [
            individual for individual, _ in population_with_scores]

        # culling
        num_to_remove = int(
            len(population_with_scores_reshaped) * cull_percent / 100)
        population = population_with_scores_reshaped[:len(
            population_with_scores_reshaped) - num_to_remove]

        # fitness_scores = [calculate_fitness(individual) for individual in population]
        # fitness_mean_log.append(np.mean(fitness_scores))

        # adds new pop after culling
        while not len(population) >= population_size:
            population.append(generate_individual(problem))

        fitness_scores = [calculate_fitness(
            individual) for individual in population]

        highest_fitness = max(fitness_scores)
        max_fitness = max(highest_fitness, max_fitness)
        best_individual = population[fitness_scores.index(highest_fitness)]

        # logging
        fittest_individual_log.append(best_individual)
        fittest_score_log.append([highest_fitness, generation])
        # if generations % generations // 10 == 0:
        #   mutation_rate, is_stuck = dynamic_mutation_rates(fitness_mean_log, 10, mutation_rate)
        #   if (is_stuck):
        #     population = [
        #         population[i] for i in range(len(population))
        #         if i >= population_size or random.random() > 0.95
        #     ]
        #     while (len(population) != population_size):
        #       population.append(generate_individual(n))
        #     fitness_scores = [calculate_fitness(individual) for individual in population]

        if (highest_fitness > best_so_far):
            best_so_far = highest_fitness
            print(f"{best_so_far}", end=", ")
            gens_without_improve = 0
        else:
            gens_without_improve += 1
        
        if (highest_fitness == 162):
            print("Found solution in generation " + str(generation))
            success = True
            break
        elif (gens_without_improve == restart_gen):
            restart_num += 1
            print(f"Restarting, {restart_num} restarts so far")
            break

    plt.clf()
    # Separate the data into two lists for plotting
    y_values = np.array([point[0] for point in fittest_score_log])
    x_values = np.array([point[1] for point in fittest_score_log])

    # Create the scatter plot
    plt.plot(x_values, y_values, color='blue', label='Fitness Scores')

    # Add labels and title
    plt.ylabel('Highest Fitness')
    plt.xlabel('Generations')
    title_string = f"{n}x{n}, mutation rate: {mutation_rate}, pop size: {population_size}, max: {highest_fitness}"
    plt.title(title_string, color='green' if success else 'red')

    # Add legend
    plt.legend()

    # Get the current time to create a folder
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    plt.savefig(current_time + '.png')
    best_individual = population[fitness_scores.index(max(fitness_scores))]
    return best_individual, generation, success

def get_possible_values(board, row, col):
    """Return the set of possible values for a given empty cell at (row, col)"""
    # Possible values are the numbers 1-9
    all_values = set(range(1, 10))
    
    # Eliminate numbers already present in the row, column, and 3x3 subgrid
    row_values = set(board[row, :]) - {0}
    col_values = set(board[:, col]) - {0}
    # Determine the top-left corner of the subgrid
    subgrid_row_start = (row // 3) * 3
    subgrid_col_start = (col // 3) * 3
    subgrid_values = set(board[subgrid_row_start:subgrid_row_start+3, subgrid_col_start:subgrid_col_start+3].flatten()) - {0}
    
    # Remove already used numbers from all values
    possible_values = all_values - row_values - col_values - subgrid_values
    
    return possible_values

def count_variations(board):
    total_variations = 1
    
    # Go through each subgrid and count the possible variations for the empty spaces
    for i in range(3):  # 3 rows of subgrids
        for j in range(3):  # 3 columns of subgrids
            subgrid_cells = []
            # Extract the cells in the 3x3 subgrid
            for r in range(i * 3, (i + 1) * 3):
                for c in range(j * 3, (j + 1) * 3):
                    if board[r, c] == 0:  # Empty space
                        subgrid_cells.append((r, c))
            
            # For each empty cell in the subgrid, get the possible values
            subgrid_possibilities = []
            for r, c in subgrid_cells:
                possible_values = get_possible_values(board, r, c)
                subgrid_possibilities.append(possible_values)
            
            # Calculate the number of possible combinations for the subgrid
            if subgrid_possibilities:
                subgrid_combinations = 1
                for possible_values in subgrid_possibilities:
                    subgrid_combinations *= len(possible_values)
                total_variations *= subgrid_combinations
    
    return total_variations

n = 9
population_size = 500
generations = 200
mutation_rate = 0.25
test_results = []
successes = 0
max_mutations = 1
cull_rate = 75
restart_gen = 60

min_gen = 0
max_gen = 0
iterations = 1000
total_time = 0
for i in range(iterations):
    print("Training iteration " + str(i + 1) + ", number of successes so far: " + str(successes) + " out of " + str(i))
    # Generate a Sudoku puzzle with a random difficulty
    puzzle = Sudoku(3, seed=time.time()).difficulty(0.5)  #np.random.rand() Difficulty levels are "easy", "medium", "hard"
    puzzle.show()
    # Get the puzzle grid (it will be a 2D list)
    grid = puzzle.board
    grid = [[0 if cell is None else cell for cell in row] for row in grid]
    grid = np.array(grid)
    max_population_size = count_variations(grid)
    # Convert the 2D grid into a 1D NumPy array (flatten it)
    PROBLEM = np.array(grid).flatten()
    
    # START TIME
    start_time = time.time()
    solution, gens_to_conv, success = genetic_algorithm(PROBLEM, n, population_size, generations, mutation_rate, max_mutations, cull_rate, restart_gen)
    # STOP TIME
    iteration_time = time.time() - start_time
    total_time += iteration_time  # Accumulate total time
    # Calculate average time per iteration
    avg_time_per_iteration = total_time / (i + 1)
    
    if success:
        total_time += iteration_time  # Accumulate total time
        # Calculate average time per iteration
        avg_time_per_iteration = total_time / (i + 1)
        min_gen = min(gens_to_conv, min_gen)
        max_gen = max(gens_to_conv, max_gen)
        print_sudoku(solution)
        successes += 1
        test_results.append(gens_to_conv)
    print(f"{np.mean(np.array(test_results))} generations average for {iterations} iterations\n max gens: {max_gen}, min gens {min_gen}, average time {avg_time_per_iteration}")


print(str(iterations) + " training iterations\n" + str(len(test_results)) + "total successes (" + str(iterations//len(test_results)) + ")")
