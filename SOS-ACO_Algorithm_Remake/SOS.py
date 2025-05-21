import numpy as np
from parameter import *
from ACO import *
from typing import Union, Tuple

class Organism(object):
    def __init__(self, phenotypes: np.ndarray) -> None:
        """
        Organism in Symbiotic Optimization Search.
        :param phenotypes: Array of attributes [alpha, beta].
        """
        self.phenotypes = phenotypes
        self.fitness = None  # Fitness computed only when needed
        self.ACO = None  # ACO instance set when fitness is computed

    def compute_fitness(self, towns: np.ndarray) -> None:
        """
        Compute fitness using ACO if not already computed.
        :param towns: Coordinates of the cities for TSP.
        """
        if self.fitness is None:
            self.ACO = ACO(
                ants=ANTS,
                evaporation_rate=EVAPORATION_RATE,
                intensification=INTENSIFICATION,
                alpha=self.phenotypes[0],
                beta=self.phenotypes[1],
                beta_evaporation_rate=0.005
            )
            self.ACO.fit(towns, iterations=25, conv_crit=20, verbose=False)
            self.fitness = 1 / self.ACO.best

    def __str__(self):
        return f'Phenotypes: {self.phenotypes}, Fitness: {self.fitness}'

class SOS(object):
    def __init__(self, lower_bound: float, upper_bound: float, population_size: int, fitness_size: int, ants: int) -> None:
        """
        Symbiotic Optimization Search to find optimal alpha and beta for ACO in TSP.
        :param lower_bound: Lower bound of parameter values.
        :param upper_bound: Upper bound of parameter values.
        :param population_size: Number of organisms in population.
        :param fitness_size: Number of parameters (2 for alpha, beta).
        :param ants: Number of ants for ACO.
        """
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.population_size = population_size
        self.fitness_size = fitness_size
        self.ants = ants
        self.population = None
        self.best_organism = None
        self.towns = TOWNS  # City coordinates for TSP
        self.fitness_cache = {}  # Cache to store fitness for phenotypes

    def random_parameters(self, a: float, b: float, size: Union[int, Tuple[int, int], None] = None) -> Union[float, np.ndarray]:
        """
        Generate random parameters within bounds.
        :param a: Lower bound.
        :param b: Upper bound.
        :param size: Output size (None for scalar, int for 1D array, tuple for 2D array).
        :return: Random number(s) between a and b.
        """
        if size is None:
            return float(np.random.uniform(low=a, high=b, size=1)[0])
        return np.random.uniform(low=a, high=b, size=size)

    def gen_population(self) -> None:
        """
        Initialize population of organisms and compute fitness efficiently.
        """
        print(f"Initializing {self.population_size} organisms...")
        population_params = self.random_parameters(self.lower_bound, self.upper_bound, (self.population_size, self.fitness_size))
        self.population = np.array([Organism(p) for p in population_params], dtype=object)

        # Compute fitness for all organisms, using cache to avoid redundant ACO runs
        for organism in self.population:
            phenotype_tuple = tuple(organism.phenotypes)
            if phenotype_tuple in self.fitness_cache:
                organism.fitness = self.fitness_cache[phenotype_tuple]
                organism.ACO = ACO(
                    ants=ANTS,
                    evaporation_rate=EVAPORATION_RATE,
                    intensification=INTENSIFICATION,
                    alpha=organism.phenotypes[0],
                    beta=organism.phenotypes[1],
                    beta_evaporation_rate=0.005
                )
                organism.ACO.fit(self.towns, iterations=25, conv_crit=20, verbose=False)
            else:
                organism.compute_fitness(self.towns)
                self.fitness_cache[phenotype_tuple] = organism.fitness

        self.set_best_organism()

    def create_new_organism(self, attributes: np.ndarray) -> Organism:
        """
        Create a new organism with given attributes.
        :param attributes: Array of attributes [alpha, beta].
        :return: New organism.
        """
        new_organism = Organism(attributes)
        phenotype_tuple = tuple(attributes)
        if phenotype_tuple in self.fitness_cache:
            new_organism.fitness = self.fitness_cache[phenotype_tuple]
            new_organism.ACO = ACO(
                ants=ANTS,
                evaporation_rate=EVAPORATION_RATE,
                intensification=INTENSIFICATION,
                alpha=attributes[0],
                beta=attributes[1],
                beta_evaporation_rate=0.005
            )
            new_organism.ACO.fit(self.towns, iterations=25, conv_crit=20, verbose=False)
        else:
            new_organism.compute_fitness(self.towns)
            self.fitness_cache[phenotype_tuple] = new_organism.fitness
        return new_organism

    def mutualism(self, i: int) -> None:
        """
        Mutualism phase with one ACO run for new_Xi if necessary.
        :param i: Index of organism Xi.
        """
        indices = np.arange(self.population_size)
        mask = indices != i
        available_indices = indices[mask]
        j = np.random.choice(available_indices)

        Xi = self.population[i]
        Xj = self.population[j]

        bf1 = np.random.randint(1, 3)
        array_rand = self.random_parameters(0, 1, self.fitness_size)
        mutual_vector = (Xi.phenotypes + Xj.phenotypes) / 2
        new_Xi_params = Xi.phenotypes + array_rand * (self.best_organism.phenotypes - mutual_vector * bf1)
        new_Xi_params = np.clip(new_Xi_params, self.lower_bound, self.upper_bound)

        new_Xi = self.create_new_organism(new_Xi_params)
        if new_Xi.fitness > Xi.fitness:
            self.population[i] = new_Xi

    def commensalism(self, i: int) -> None:
        """
        Commensalism phase with one ACO run if necessary.
        :param i: Index of organism Xi.
        """
        indices = np.arange(self.population_size)
        mask = indices != i
        available_indices = indices[mask]
        j = np.random.choice(available_indices)

        Xi = self.population[i]
        Xj = self.population[j]

        array_rand = self.random_parameters(-1, 1, self.fitness_size)
        new_Xi_params = Xi.phenotypes + array_rand * (self.best_organism.phenotypes - Xj.phenotypes)
        new_Xi_params = np.clip(new_Xi_params, self.lower_bound, self.upper_bound)

        new_Xi = self.create_new_organism(new_Xi_params)
        if new_Xi.fitness > Xi.fitness:
            self.population[i] = new_Xi

    def parasitism(self, i: int) -> None:
        """
        Parasitism phase with one ACO run.
        :param i: Index of organism Xi.
        """
        parasite_params = np.copy(self.population[i].phenotypes)
        indices = np.arange(self.population_size)
        mask = indices != i
        available_indices = indices[mask]
        j = np.random.choice(available_indices)

        Xj = self.population[j]
        dim_index = np.random.randint(0, self.fitness_size)
        parasite_params[dim_index] = self.random_parameters(self.lower_bound, self.upper_bound)

        parasite = self.create_new_organism(parasite_params)
        if parasite.fitness > Xj.fitness:
            self.population[j] = parasite

    def excute_sos(self) -> None:
        """
        Execute SOS algorithm with minimal ACO runs.
        """
        print("Executing SOS algorithm...")
        for i in range(self.population_size):
            self.mutualism(i)
            self.commensalism(i)
            self.parasitism(i)
            self.set_best_organism()
        print("Finished SOS algorithm")

    def set_best_organism(self) -> None:
        """
        Set the best organism from the population.
        """
        fitness_values = np.array([organism.fitness for organism in self.population])
        best_index = np.argmax(fitness_values)
        self.best_organism = self.population[best_index]