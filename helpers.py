import numpy as np
from scipy import stats
import csv

def get_stats_string(data):
  # Extract the generations (second column of the input data)
  generations = [row[1] for row in data]

  # Calculate mean, median, variance, standard deviation, min, max, range
  mean_gen = np.mean(generations)
  median_gen = np.median(generations)
  variance_gen = np.var(generations)
  std_dev_gen = np.std(generations)
  min_gen = np.min(generations)
  max_gen = np.max(generations)
  range_gen = max_gen - min_gen

  # Calculate skewness and kurtosis
  skewness_gen = stats.skew(generations)
  kurtosis_gen = stats.kurtosis(generations)

  # Accumulate all statistics in a string
  output = ""
  output += f"Mean Generations: {mean_gen}\n"
  output += f"Median Generations: {median_gen}\n"
  output += f"Variance of Generations: {variance_gen}\n"
  output += f"Standard Deviation of Generations: {std_dev_gen}\n"
  output += f"Minimum Generations: {min_gen}\n"
  output += f"Maximum Generations: {max_gen}\n"
  output += f"Range of Generations: {range_gen}\n"
  output += f"Skewness of Generations: {skewness_gen}\n"
  output += f"Kurtosis of Generations: {kurtosis_gen}\n"

  return output


def write_to_csv_stats(population, fitness_scores, name):
  # sorted_fitness_scores = sorted(fitness_scores, reverse=True)

  # Determine the maximum width for formatting numbers (for right alignment)
  # We look for the largest number in the population for the width
  max_width = max(len(str(num))
    for individual in population for row in individual for num in row)

  with open(name + ".csv", mode="w", newline="") as file:
    # Create the CSV writer and write the header row
    writer = csv.writer(file)
    writer.writerow(["Individual", "Generations"])

    for individual, fitness in zip(population, fitness_scores):
      # Format the matrix (individual) so that each number is right-aligned
      formatted_individual = "\n".join([" ".join(f"{num:>{max_width}}" for num in row)for row in individual])

      # Format fitness as a string (assuming it's a tuple with score and generations)
      formatted_fitness = f"Score: {
        fitness[0]}, Generations: {fitness[1]}"

      # Write to CSV
      writer.writerow([formatted_individual, formatted_fitness])
