import random
import math
import sys
import sudoku
# Sudoku size and grid
SIZE = 9
SUBGRIDS = 3

# Count the number of conflicts in the grid
def count_conflicts(grid):
    conflicts = 0
    
    # Row conflicts
    for row in grid:
        counts = [0] * (SIZE + 1)
        for num in row:
            if num != 0:
                counts[num] += 1
        conflicts += sum(c - 1 for c in counts if c > 1)
    
    # Column conflicts
    for col in range(SIZE):
        counts = [0] * (SIZE + 1)
        for row in range(SIZE):
            num = grid[row][col]
            if num != 0:
                counts[num] += 1
        conflicts += sum(c - 1 for c in counts if c > 1)
    
    # Subgrid conflicts
    for box_row in range(0, SIZE, SUBGRIDS):
        for box_col in range(0, SIZE, SUBGRIDS):
            counts = [0] * (SIZE + 1)
            for i in range(SUBGRIDS):
                for j in range(SUBGRIDS):
                    num = grid[box_row + i][box_col + j]
                    if num != 0:
                        counts[num] += 1
            conflicts += sum(c - 1 for c in counts if c > 1)
    
    return conflicts

# Get a random neighbor (change a number in the grid)
def get_neighbor(grid):
    new_grid = [row[:] for row in grid]
    empty_cells = [(r, c) for r in range(len(grid)) for c in range(len(grid[r])) if grid[r][c] == 0]
    
    # If no empty cells are found, return the original grid
    if not empty_cells:
        return new_grid
    
    # Randomly select an empty cell
    row, col = random.choice(empty_cells)
    
    # Assign a random number (1-9 for Sudoku)
    new_grid[row][col] = random.randint(1, len(grid))
    return new_grid

# Simulated Annealing algorithm to solve Sudoku
def simulated_annealing(grid, max_iterations=100000, initial_temp=100, temp_decay=0.995):
    current_grid = grid
    current_conflicts = count_conflicts(current_grid)
    
    temp = initial_temp
    best_grid = current_grid
    best_conflicts = current_conflicts
    
    while True:
    #for _ in range(max_iterations):
        # Generate a neighbor solution
        neighbor = get_neighbor(current_grid)
        neighbor_conflicts = count_conflicts(neighbor)
        
        # If the neighbor has fewer conflicts, accept it
        if neighbor_conflicts < current_conflicts:
            current_grid = neighbor
            current_conflicts = neighbor_conflicts
            
            if current_conflicts < best_conflicts:
                best_grid = current_grid
                best_conflicts = current_conflicts
        else:
            # Accept the neighbor with a probability
            acceptance_prob = math.exp((current_conflicts - neighbor_conflicts) / temp)
            if random.random() < acceptance_prob:
                current_grid = neighbor
                current_conflicts = neighbor_conflicts
        
        # Cooling schedule
        temp *= temp_decay
        
        # If the solution is optimal, exit early
        if current_conflicts == 0:
            break
    
    return best_grid, current_grid, best_conflicts

# Main execution
if __name__ == "__main__":
    initial_board = [
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9],
    ]
    initial_grid = sudoku.Sudoku(initial_board)
    print("Initial Sudoku Grid:")
    
    solved_grid, last_grid, conflicts = simulated_annealing(initial_board)
    print("\nSolved Sudoku Grid:")
    for i in range(9):
        if i % 3 == 0 and i != 0:
            print("-" * 21)  # Horizontal separator for 3x3 grids
        for j in range(9):
            if j % 3 == 0 and j != 0:
                print("| ", end="")  # Vertical separator for 3x3 grids
            print(last_grid[i][j] if last_grid[i][j] != 0 else ".", end=" ")
        print()
    print(f"\nFinal conflicts: {conflicts}")
