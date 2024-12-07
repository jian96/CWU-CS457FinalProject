import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    Flatten,
    Dense,
    Reshape,
    BatchNormalization,
    ReLU,
)
from tensorflow.keras.optimizers import Adam
from sudoku import Sudoku

def generate_sudoku_dataset(num_samples):
    """
    Generates a dataset of Sudoku puzzles and their solutions.
    :param num_samples: Number of puzzles to generate.
    :return: A tuple of (puzzles, solutions) where both are lists of 9x9 grids.
    """
    puzzles = []
    solutions = []

    for _ in range(num_samples):
        # Generate a random Sudoku puzzle
        sudoku = Sudoku(3).difficulty(random.uniform(0, 1))  # 3x3 blocks -> 9x9 grid
        puzzle = sudoku.board  # Get the unsolved puzzle (2D list)
        solution = sudoku.solve().board  # Get the solved puzzle (2D list)

        puzzles.append(puzzle)
        solutions.append(solution)

    return np.array(puzzles), np.array(solutions)

def preprocess_data(puzzles, solutions):
    """
    Preprocess the Sudoku data.
    :param puzzles: List of Sudoku boards (9x9 grids) as input.
    :param solutions: Corresponding solved Sudoku boards (9x9 grids).
    :return: Preprocessed input and label data.
    """
    # Normalize input puzzles to [0, 1]
    puzzles = np.array(puzzles).astype('float32') / 9.0
    puzzles = np.expand_dims(puzzles, axis=-1)  # Add channel dimension

    # Convert solutions to one-hot encoding
    solutions = np.array(solutions) - 1  # Adjust range to [0, 8]
    solutions = np.expand_dims(solutions, axis=-1)  # Add channel dimension

    return puzzles, solutions

# Example usage:
# puzzles, solutions = preprocess_data(raw_puzzles, raw_solutions)
# X_train, y_train = puzzles[:split], solutions[:split]
# X_test, y_test = puzzles[split:], solutions[split:]

# Define the CNN model
def build_sudoku_cnn():
    input_shape = (9, 9, 1)  # Grayscale image with one channel
    num_classes = 9  # Numbers 1-9

    # Input layer
    inputs = Input(shape=input_shape)

    # Convolutional layers
    x = Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu')(inputs)
    x = BatchNormalization()(x)

    x = Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)

    x = Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)

    # Flatten and fully connected layers
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(512, activation='relu')(x)

    # Output layer: 81 cells, each with 9 possible outputs
    x = Dense(81 * num_classes, activation='softmax')(x)
    outputs = Reshape((9, 9, num_classes))(x)

    model = Model(inputs, outputs)
    return model

def solve_sudoku(model, puzzle):
    """
    Solve a single Sudoku puzzle using the trained model.
    :param model: Trained Sudoku CNN model.
    :param puzzle: Input Sudoku puzzle (9x9 array).
    :return: Solved Sudoku puzzle.
    """
    puzzle = np.expand_dims(puzzle, axis=0)  # Add batch dimension
    prediction = model.predict(puzzle)
    solved_board = np.argmax(prediction, axis=-1) + 1  # Decode one-hot encoding
    return solved_board[0]  # Remove batch dimension

# Generate 10,000 samples
raw_puzzles, raw_solutions = generate_sudoku_dataset(10000)
print(f"Generated {len(raw_puzzles)} puzzles and {len(raw_solutions)} solutions.")

# Example usage:
# solved_board = solve_sudoku(model, X_test[0])

# Preprocess data
X, y = preprocess_data(raw_puzzles, raw_solutions)

# Split into training and testing datasets
split = int(len(X) * 0.8)  # 80% train, 20% test
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"Training data shape: X_train={X_train.shape}, y_train={y_train.shape}")
print(f"Testing data shape: X_test={X_test.shape}, y_test={y_test.shape}")

# Compile the model
model = build_sudoku_cnn()
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Summary of the model
model.summary() 

# Train the model
history = model.fit(X_train, y_train,
                    validation_data=(X_test, y_test),
                    batch_size=32,
                    epochs=20)