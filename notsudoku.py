import random

class Sudoku:
    """
    A class to represent and manipulate a Sudoku puzzle.
    """

    def __init__(self, board: list[list[int]]):
        """
        Initialize the Sudoku board.
        :param board: A 9x9 list of lists representing the Sudoku puzzle.
                      Empty cells are represented by 0.
        """
        self.board = board
        
    def init_solution_row(self):
        """Generate a random solution from a Sudoku problem
        """
        solution = []
        for i in range(0, 81, 9):
            row = self.board[i:i+9]
            permu = [n for n in range(1,10) if n not in row]
            random.shuffle(permu)
            solution.extend([n or permu.pop() for n in row])
        self.board = solution
        return solution

    def is_valid_move(self, row: int, col: int, num: int) -> bool:
        """
        Check if placing `num` at position (row, col) is valid.
        """
        # Check the row
        if num in self.board[row]:
            return False

        # Check the column
        for i in range(9):
            if self.board[i][col] == num:
                return False

        # Check the 3x3 subgrid
        start_row, start_col = (row // 3) * 3, (col // 3) * 3
        for i in range(start_row, start_row + 3):
            for j in range(start_col, start_col + 3):
                if self.board[i][j] == num:
                    return False

        return True

    def print_board(self) -> None:
        """
        Print the Sudoku board in a user-friendly format.
        """
        for i in range(9):
            if i % 3 == 0 and i != 0:
                print("-" * 21)  # Horizontal separator for 3x3 grids
            for j in range(9):
                if j % 3 == 0 and j != 0:
                    print("| ", end="")  # Vertical separator for 3x3 grids
                print(self.board[i][j] if self.board[i][j] != 0 else ".", end=" ")
            print()

    def solve(self) -> bool:
        """
        Solve the Sudoku puzzle using backtracking.
        :return: True if a solution exists, False otherwise.
        """
        empty_cell = self.find_empty_cell()
        if not empty_cell:
            return True  # Puzzle is solved

        row, col = empty_cell

        for num in range(1, 10):  # Try numbers 1 through 9
            if self.is_valid_move(row, col, num):
                self.board[row][col] = num

                if self.solve():  # Recursively attempt to solve
                    return True

                self.board[row][col] = 0  # Backtrack

        return False

    def find_empty_cell(self) -> tuple[int, int] | None:
        """
        Find the next empty cell in the Sudoku board.
        :return: A tuple (row, col) of the empty cell's position, or None if no empty cells remain.
        """
        for i in range(9):
            for j in range(9):
                if self.board[i][j] == 0:
                    return i, j
        return None


# Example usage
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

    sudoku = Sudoku(initial_board)
    print("Initial Board:")
    sudoku.init_solution_row()
    sudoku.print_board()

    if sudoku.solve():
        print("\nSolved Board:")
        sudoku.print_board()
    else:
        print("\nNo solution exists.")
