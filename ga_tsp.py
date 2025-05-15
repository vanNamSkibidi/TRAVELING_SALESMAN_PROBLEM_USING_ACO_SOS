import time
import random
from typing import List, Tuple
import math
from parameter import *

# Reuse utility functions from the provided code
def euclidean_distance(point1: list, point2: list) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(point1, point2)))

def pairwise_distances(points1: list, points2: list) -> List[List[float]]:
    distances = []
    for p1 in points1:
        row = []
        for p2 in points2:
            row.append(euclidean_distance(p1, p2))
        distances.append(row)
    return distances

def round_matrix(matrix: list, afterDot: int) -> List[List[float]]:
    return [[round(x, afterDot) for x in row] for row in matrix]

def read_map(file_path: str) -> List[List[float]]:
    with open(file_path, "r") as file:
        cities = []
        city_flag = False
        for line in file:
            if line.strip() == "NODE_COORD_SECTION":
                city_flag = True
                continue
            if line.strip() == "EOF":
                return cities 
            if city_flag:
                city_coor = [float(i) for i in line.strip().split()[1:]]
                cities.append(city_coor)

# Parameters from parameter.py (assumed similar to the original code)
PATH_TO_MAP = "./Benchmark/eil51.tsp"
TOWNS = read_map(PATH_TO_MAP)

class GeneticAlgorithmTSP:
    def __init__(self, population_size: int, tournament_size: int, crossover_rate: float, 
                 mutation_rate: float, elitism_count: int) -> None:
        """
        Genetic Algorithm for solving the Traveling Salesman Problem (TSP).

        :param population_size: Number of individuals in the population.
        :param tournament_size: Number of individuals in tournament selection.
        :param crossover_rate: Probability of performing crossover.
        :param mutation_rate: Probability of mutating each gene.
        :param elitism_count: Number of best individuals to carry over to the next generation.
        """
        self.population_size = population_size
        self.tournament_size = tournament_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism_count = elitism_count

        self.distance_matrix = None
        self.spatial_map = None
        self.num_nodes = None
        self.population = None

        # Internal statistics
        self.best_path = None
        self.best_distance = None
        self.best_series = None
        self.global_best = None
        self.fit_time = None
        self.fitted = False
        self.converged = False
        self.stopped_at_iteration = None

    def __str__(self) -> str:
        """Get information of GA in string format"""
        string = "Genetic Algorithm for TSP"
        string += "\n------------------------"
        string += "\nDesigned to solve the traveling salesman problem by evolving city permutations."
        string += "\n------------------------"
        string += f"\nPopulation size:\t\t\t{self.population_size}"
        string += f"\nTournament size:\t\t\t{self.tournament_size}"
        string += f"\nCrossover rate:\t\t\t{self.crossover_rate}"
        string += f"\nMutation rate:\t\t\t{self.mutation_rate}"
        string += f"\nElitism count:\t\t\t{self.elitism_count}"
        string += "\n------------------------"
        if self.fitted:
            string += "\n\nThis optimizer has been fitted."
        else:
            string += "\n\nThis optimizer has NOT been fitted."
        return string

    def __initialize(self, spatial_map: List[List[float]]) -> None:
        """
        Initialize the distance matrix and population.

        :param spatial_map: List of coordinates of cities in 2D plane. Ex: [[x1,y1]...,[xn,yn]]
        """
        for point in spatial_map:
            assert len(point) == 2, "Invalid coordinates! Each point must have x, y values."
        self.spatial_map = spatial_map
        self.num_nodes = len(spatial_map)
        self.distance_matrix = round_matrix(pairwise_distances(spatial_map, spatial_map), 0)

        # Initialize population with random permutations
        self.population = []
        for _ in range(self.population_size):
            path = list(range(self.num_nodes))
            random.shuffle(path)
            self.population.append(path)

    def __calculate_distance(self, path: List[int]) -> float:
        """
        Calculate the total distance of a TSP tour.

        :param path: List of city indices representing the tour.
        :return: Total distance of the tour.
        """
        distance = 0.0
        for i in range(len(path) - 1):
            distance += self.distance_matrix[path[i]][path[i + 1]]
        distance += self.distance_matrix[path[-1]][path[0]]  # Return to start
        return distance

    def __fitness(self, path: List[int]) -> float:
        """
        Calculate fitness as the inverse of the tour distance.

        :param path: List of city indices.
        :return: Fitness value (higher is better).
        """
        distance = self.__calculate_distance(path)
        return 1.0 / distance if distance > 0 else float('inf')

    def __tournament_selection(self) -> List[int]:
        """
        Perform tournament selection to choose a parent.

        :return: Selected individual (tour).
        """
        tournament = random.sample(self.population, self.tournament_size)
        return max(tournament, key=self.__fitness)

    def __ordered_crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        """
        Perform ordered crossover (OX) to produce an offspring.

        :param parent1: First parent tour.
        :param parent2: Second parent tour.
        :return: Offspring tour.
        """
        if random.random() > self.crossover_rate:
            return parent1.copy()  # No crossover

        size = len(parent1)
        start, end = sorted(random.sample(range(size), 2))
        offspring = [None] * size
        # Copy segment from parent1
        offspring[start:end + 1] = parent1[start:end + 1]
        # Fill remaining positions with parent2's cities in order
        p2_index = 0
        for i in range(size):
            if start <= i <= end:
                continue
            while parent2[p2_index] in offspring:
                p2_index += 1
            offspring[i] = parent2[p2_index]
        return offspring

    def __swap_mutation(self, path: List[int]) -> List[int]:
        """
        Perform swap mutation on a tour.

        :param path: Tour to mutate.
        :return: Mutated tour.
        """
        path = path.copy()
        for i in range(len(path)):
            if random.random() < self.mutation_rate:
                j = random.randint(0, len(path) - 1)
                path[i], path[j] = path[j], path[i]
        return path

    def fit(self, spatial_map: List[List[float]], generations: int, conv_crit: int = 20, 
            verbose: bool = True, decimal: int = 2) -> None:
        """
        Run the GA to optimize the TSP.

        :param spatial_map: List of city coordinates.
        :param generations: Maximum number of generations.
        :param conv_crit: Number of generations with no improvement to stop.
        :param verbose: Print progress if True.
        :param decimal: Decimal places for output.
        """
        start = time.time()
        self.__initialize(spatial_map)
        num_equal = 0
        self.best_series = [0.0] * generations
        self.global_best = [0.0] * generations

        if verbose:
            print(f"{self.num_nodes} nodes given. Beginning GA Optimization with {generations} generations...\n")

        for generation in range(generations):
            start_iter = time.time()
            new_population = []

            # Elitism: Copy best individuals
            sorted_pop = sorted(self.population, key=self.__calculate_distance)
            new_population.extend(sorted_pop[:self.elitism_count])

            # Generate new individuals
            while len(new_population) < self.population_size:
                parent1 = self.__tournament_selection()
                parent2 = self.__tournament_selection()
                offspring = self.__ordered_crossover(parent1, parent2)
                offspring = self.__swap_mutation(offspring)
                new_population.append(offspring)

            self.population = new_population[:self.population_size]
            best_individual = min(self.population, key=self.__calculate_distance)
            best_distance = self.__calculate_distance(best_individual)

            if generation == 0 or best_distance < (self.best_distance or float('inf')):
                self.best_distance = best_distance
                self.best_path = best_individual
                num_equal = 0
            else:
                num_equal += 1

            self.best_series[generation] = best_distance
            self.global_best[generation] = self.best_distance

            if verbose:
                print(
                    f"Generation {generation}/{generations} | Distance: {round(best_distance, decimal)} | "
                    f"Best so far: {round(self.best_distance, decimal)} | "
                    f"{round(time.time() - start_iter, decimal)} s | num_equal: {num_equal}"
                )

            if num_equal >= conv_crit:
                self.converged = True
                self.stopped_at_iteration = generation
                if verbose:
                    print("\nConvergence criterion met. Stopping...")
                break

        if not self.converged:
            self.stopped_at_iteration = generations
        self.fit_time = round(time.time() - start)
        self.fitted = True

        if self.converged:
            self.best_series = [x for x in self.best_series if x > 0]

        if verbose:
            print(
                f"\nGA fitted. Runtime: {self.fit_time // 60} minute(s). Best distance: {round(self.best_distance, decimal)}"
            )

    def get_result(self) -> Tuple[List[int], float, float, List[float]]:
        """
        Return the optimization results.

        :return: Tuple of (best path, best distance, fit time, best distances per generation).
        """
        return self.best_path, self.best_distance, self.fit_time, self.best_series

def main():
    """
    Main function to run the GA for TSP.
    """
    ga = GeneticAlgorithmTSP(
        population_size=50,  # Number of tours in population
        tournament_size=5,   # Size of tournament for selection
        crossover_rate=0.8,  # Probability of crossover
        mutation_rate=0.1,   # Probability of mutation per gene
        elitism_count=2      # Number of best tours to preserve
    )
    ga.fit(spatial_map=TOWNS, generations=200, conv_crit=25, verbose=True)
    best_path, best_distance, fit_time, best_series = ga.get_result()
    best_path_coor = [TOWNS[i] for i in best_path]
    print("Best path coordinates:", best_path_coor)
    print("Best distance:", round(best_distance, 2))

if __name__ == "__main__":
    main()