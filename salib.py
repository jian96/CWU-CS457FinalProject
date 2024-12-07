import copy
import random
import numpy as np
from simanneal import Annealer
from sudoku import Sudoku
import time

# from https://neos-guide.org/content/sudoku
_ = 0
# PROBLEM = [
#     1, _, _,  _, _, 6,  3, _, 8,
#     _, _, 2,  3, _, _,  _, 9, _,
#     _, _, _,  _, _, _,  7, 1, 6,

#     7, _, 8,  9, 4, _,  _, _, 2,
#     _, _, 4,  _, _, _,  9, _, _,
#     9, _, _,  _, 2, 5,  1, _, 4,

#     6, 2, 9,  _, _, _,  _, _, _,
#     _, 4, _,  _, _, 7,  6, _, _,
#     5, _, 7,  6, _, _,  _, _, 3,
# ]

PROBLEM = np.array([
    5, 3, 0, 0, 7, 0, 0, 0, 0,
    6, 0, 0, 1, 9, 5, 0, 0, 0,
    0, 9, 8, 0, 0, 0, 0, 6, 0,
    8, 0, 0, 0, 6, 0, 0, 0, 3,
    4, 0, 0, 8, 0, 3, 0, 0, 1,
    7, 0, 0, 0, 2, 0, 0, 0, 6,
    0, 6, 0, 0, 0, 0, 2, 8, 0,
    0, 0, 0, 4, 1, 9, 0, 0, 5,
    0, 0, 0, 0, 8, 0, 0, 7, 9,
])

# PROBLEM = np.array([
#     0, 0, 0, 7, 0, 0, 0, 0, 0,
#     8, 4, 0, 0, 0, 0, 6, 3, 5,
#     3, 1, 0, 0, 5, 0, 0, 7, 2,
#     0, 0, 0, 3, 0, 8, 1, 9, 6,
#     0, 0, 0, 0, 9, 0, 2, 5, 4,
#     9, 6, 0, 2, 0, 5, 7, 0, 3,
#     0, 2, 0, 5, 8, 6, 0, 4, 9,
#     0, 5, 0, 0, 3, 2, 0, 0, 7,
#     6, 8, 0, 4, 0, 9, 0, 0, 1,
# ])

def print_sudoku(state):
    border = "------+-------+------"
    rows = [state[i:i+9] for i in range(0,81,9)]
    for i,row in enumerate(rows):
        if i % 3 == 0:
            print(border)
        three = [row[i:i+3] for i in range(0,9,3)]
        print(" | ".join(
            " ".join(str(x or "_") for x in one)
            for one in three
        ))
    print(border)

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

def initial_solution(problem):
    """provide sudoku problem, generate an init solution by randomly filling
    each sq block without considering row/col consistency"""
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

class Sudoku_Sq(Annealer):
    def __init__(self, problem):
        self.problem = problem
        state = initial_solution(problem)
        super().__init__(state)
    def move(self):
        """randomly swap two cells in a random square"""
        block = random.randrange(9)
        indices = [i for i in block_indices(block) if self.problem[i] == 0]
        if len(indices) >= 2:
            m, n = random.sample(indices, 2)
            self.state[m], self.state[n] = self.state[n], self.state[m]
        else: 
            print("Not enough elements to sample.")
            # Handle the situation where sampling isn't possible, e.g., skip or raise an error.
    def energy(self):
        """calculate the number of violations: assume all rows are OK"""
        column_score = lambda n: -len(set(self.state[coord(i, n)] for i in range(9)))
        row_score = lambda n: -len(set(self.state[coord(n, i)] for i in range(9)))
        score = sum(column_score(n)+row_score(n) for n in range(9))
        if score == -162:
            self.user_exit = True # early quit, we found a solution
        return score

def main():
    iterations = 500
    total_time = 0
    for i in range(iterations):
        print("Training iteration " + str(i + 1))
        random.seed(time.time())
        # Generate a Sudoku puzzle with a random difficulty
        puzzle = Sudoku(3, seed=time.time()).difficulty(0.5)  #np.random.rand() Difficulty levels are "easy", "medium", "hard"
        puzzle.show()
        # Get the puzzle grid (it will be a 2D list)
        grid = puzzle.board
        grid = [[0 if cell is None else cell for cell in row] for row in grid]
        grid = np.array(grid)
        # Convert the 2D grid into a 1D NumPy array (flatten it)
        PROBLEM = np.array(grid).flatten()
        
        # START TIME
        start_time = time.time()
        sudoku = Sudoku_Sq(PROBLEM)
        sudoku.copy_strategy = "method"
        print_sudoku(sudoku.state)
        sudoku.Tmax = 0.5
        sudoku.Tmin = 0.05
        sudoku.steps = 100000
        sudoku.updates = 100
        
        state, e = sudoku.anneal()
        print("\n")
        print_sudoku(state)
        print("E=%f (expect -162)" % e)
        
        # STOP TIME
        iteration_time = time.time() - start_time
        total_time += iteration_time  # Accumulate total time
        # Calculate average time per iteration
        avg_time_per_iteration = total_time / (i + 1)
    print(avg_time_per_iteration)

if __name__ == "__main__":
    main()