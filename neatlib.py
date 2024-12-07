import numpy as np
import neat
import os

# Helper function to validate if a board is a valid Sudoku board
def is_valid(board, row, col, num):
    # Check row
    if num in board[row]:
        return False
    # Check column
    if num in board[:, col]:
        return False
    # Check 3x3 subgrid
    start_row, start_col = 3 * (row // 3), 3 * (col // 3)
    for i in range(start_row, start_row + 3):
        for j in range(start_col, start_col + 3):
            if board[i][j] == num:
                return False
    return True

# Function to transform the board into inputs for the neural network
def board_to_input(board):
    return board.flatten()

# Fitness function for NEAT
def evaluate_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    # Example Sudoku puzzle (0 represents empty cells)
    puzzle = np.array([
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9]
    ])
    
    # Copy of the puzzle for solving
    board = puzzle.copy()
    fitness = 0

    # Iterate through each cell in the puzzle
    for row in range(9):
        for col in range(9):
            if board[row][col] == 0:  # Empty cell
                inputs = board_to_input(board)
                output = net.activate(inputs)
                # Predicted number (1-9, rounded)
                predicted_number = int(np.argmax(output) + 1)
                
                if is_valid(board, row, col, predicted_number):
                    board[row][col] = predicted_number
                    fitness += 1  # Increase fitness for correct prediction
                else:
                    # Penalize invalid predictions
                    fitness -= 1

    return fitness

# Configuration for NEAT
def run_neat():
    # Load NEAT configuration
    config_path = os.path.join(os.getcwd(), "config-feedforward")
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    # Create the population
    population = neat.Population(config)

    # Add reporters for output
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    # Run NEAT for a maximum of 50 generations
    winner = population.run(evaluate_genome, 50)

    # Output the winning genome
    print("\nBest genome:\n", winner)

if __name__ == "__main__":
    run_neat()
