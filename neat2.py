import random
import math
from collections import defaultdict, namedtuple

# Gene Definitions
NodeGene = namedtuple("NodeGene", ["id", "type"])  # 'type' is 'input', 'hidden', or 'output'
ConnectionGene = namedtuple("ConnectionGene", ["in_node", "out_node", "weight", "enabled", "innovation"])

class Genome:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.nodes = {}  # {node_id: NodeGene}
        self.connections = {}  # {innovation: ConnectionGene}
        self.fitness = 0.0
        self.initialize_genome()

    def initialize_genome(self):
        # Create input and output nodes
        for i in range(self.input_size):
            self.nodes[i] = NodeGene(i, "input")
        for i in range(self.output_size):
            self.nodes[self.input_size + i] = NodeGene(self.input_size + i, "output")

        # Fully connect input to output with random weights
        innovation = 0
        for i in range(self.input_size):
            for j in range(self.output_size):
                in_node = i
                out_node = self.input_size + j
                weight = random.uniform(-1, 1)
                self.connections[innovation] = ConnectionGene(in_node, out_node, weight, True, innovation)
                innovation += 1

    def mutate(self):
        """Apply mutation to the genome."""
        if random.random() < 0.8:  # Weight mutation
            self.mutate_weights()
        if random.random() < 0.1:  # Add connection mutation
            self.add_connection()
        if random.random() < 0.03:  # Add node mutation
            self.add_node()

    def mutate_weights(self):
        for conn in self.connections.values():
            if random.random() < 0.9:  # Perturb existing weight
                conn.weight += random.uniform(-0.5, 0.5)
            else:  # Assign new random weight
                conn.weight = random.uniform(-1, 1)

    def add_connection(self):
        # Add a random connection between existing nodes
        node_ids = list(self.nodes.keys())
        in_node, out_node = random.sample(node_ids, 2)
        if any(conn for conn in self.connections.values() if conn.in_node == in_node and conn.out_node == out_node):
            return  # Connection already exists
        innovation = len(self.connections)
        weight = random.uniform(-1, 1)
        self.connections[innovation] = ConnectionGene(in_node, out_node, weight, True, innovation)

    def add_node(self):
        # Add a new node by splitting an existing connection
        if not self.connections:
            return
        conn = random.choice(list(self.connections.values()))
        if not conn.enabled:
            return
        conn.enabled = False
        new_node_id = len(self.nodes)
        self.nodes[new_node_id] = NodeGene(new_node_id, "hidden")
        innovation1 = len(self.connections)
        innovation2 = innovation1 + 1
        self.connections[innovation1] = ConnectionGene(conn.in_node, new_node_id, 1.0, True, innovation1)
        self.connections[innovation2] = ConnectionGene(new_node_id, conn.out_node, conn.weight, True, innovation2)

class Population:
    def __init__(self, size, input_size, output_size):
        self.size = size
        self.input_size = input_size
        self.output_size = output_size
        self.genomes = [Genome(input_size, output_size) for _ in range(size)]
        self.species = defaultdict(list)

    def evaluate(self, fitness_function):
        """Evaluate all genomes and assign fitness values."""
        for genome in self.genomes:
            genome.fitness = fitness_function(genome)

    def speciate(self, threshold=3.0):
        """Group genomes into species based on similarity."""
        self.species.clear()
        for genome in self.genomes:
            for species_representative in self.species.keys():
                if self.compatibility_distance(genome, species_representative) < threshold:
                    self.species[species_representative].append(genome)
                    break
            else:  # Create new species
                self.species[genome] = [genome]

    def reproduce(self):
        """Reproduce new population based on species."""
        new_genomes = []
        for species in self.species.values():
            species.sort(key=lambda g: g.fitness, reverse=True)
            survivors = species[:max(1, len(species) // 2)]
            for _ in range(len(species)):
                parent1, parent2 = random.sample(survivors, 2)
                child = self.crossover(parent1, parent2)
                child.mutate()
                new_genomes.append(child)
        self.genomes = new_genomes

    def compatibility_distance(self, genome1, genome2):
        """Calculate the compatibility distance between two genomes."""
        disjoint = 0
        excess = 0
        weight_diff = 0
        matching = 0

        all_innovations = set(genome1.connections.keys()).union(genome2.connections.keys())
        for innovation in all_innovations:
            conn1 = genome1.connections.get(innovation)
            conn2 = genome2.connections.get(innovation)
            if conn1 and conn2:
                weight_diff += abs(conn1.weight - conn2.weight)
                matching += 1
            elif conn1 or conn2:
                disjoint += 1

        excess = abs(len(genome1.connections) - len(genome2.connections))
        N = max(len(genome1.connections), len(genome2.connections))
        N = max(N, 1)  # Avoid division by zero
        return (disjoint + excess) / N + weight_diff / matching if matching > 0 else float("inf")

    def crossover(self, parent1, parent2):
        """Create a child genome by combining genes from two parents."""
        child = Genome(self.input_size, self.output_size)
        for innovation, conn1 in parent1.connections.items():
            conn2 = parent2.connections.get(innovation)
            if conn2 and random.random() < 0.5:
                child.connections[innovation] = conn2
            else:
                child.connections[innovation] = conn1
        return child

# Example usage:
def fitness_function(genome):
    """Example fitness function."""
    return random.random()  # Replace with problem-specific evaluation

population = Population(size=100, input_size=3, output_size=1)

for generation in range(100):
    population.evaluate(fitness_function)
    population.speciate()
    population.reproduce()
    print(f"Generation {generation}: Best fitness = {max(g.fitness for g in population.genomes)}")
