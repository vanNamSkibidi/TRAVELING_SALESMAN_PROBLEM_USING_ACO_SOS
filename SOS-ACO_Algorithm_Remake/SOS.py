import numpy as np
from parameter import *
from ACO import *
from typing import Union, Tuple

class Organism(object):
    def __init__(self, phenotypes: np.ndarray) -> None:
        """
        Organism in Symbiotic optimization search
        :param phenotypes: a list of attribute of organism [alpha, beta]
        """
        self.phenotypes = phenotypes
        self.fitness = 0
        self.ACO = None
        
    def set_fitness(self, cost: float) -> None:
        """
        Calculate fitness of organism
        :param cost: the cost of solution
        """
        
        self.fitness = 1 / cost
    
    def aco_fitness(self) -> None:
        """
        Use parameters of organism for ACO
        """
        
        towns = TOWNS
        self.ACO = ACO(ants=ANTS, evaporation_rate=EVAPORATION_RATE, intensification=INTENSIFICATION, alpha=self.phenotypes[0], beta=self.phenotypes[1], beta_evaporation_rate=0.005)
        self.ACO.fit(towns, iterations=25, conv_crit=20, verbose=False)
        self.set_fitness(self.ACO.best)
    
    def __str__(self):
        return f'{self.phenotypes} = {self.fitness}'
    
class SOS(object):
    def __init__(self,
                 lower_bound: float,
                 upper_bound: float,
                 population_size: int,
                 fitness_size: int,
                 ants: int) -> None:
        """
        Symbiotic optimization search find optimal parameters for ACO
        :param lower_bound: lower bound of range value limit
        :param upper_bound: upper bound of range value limit
        :param population_size: number of organisms in population
        :param fitness_size: number of parameters contained in an organism
        :param ants: number of ants to traverse the graph
        """
        
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.population_size = population_size
        self.fitness_size = fitness_size
        self.population = None
        self.best_organism = None
        self.best_index = None
        self.ants = ants
    
    def random_parameters(self, a: float, b: float, size: Union[int, Tuple[int, int], None] = None) -> Union[float, np.ndarray]:
        """
        :param a: Lower bound (int, float).
        :param b: Upper bound (int, float).
        :param size: Size of the output:
            - None: Return a single float.
            - int: Return a 1D array of length size.
            - tuple (rows, cols): Return a 2D array of shape (rows, cols).
        :return: A single float or a NumPy array of random numbers between a and b.
        """
        
        if size is None:
            return float(np.random.uniform(low=a, high=b, size=1)[0])
    
        result = np.random.uniform(low=a, high=b, size=size)
        return result
    
    def gen_population(self) -> None:
        """
        Initialize population of organisms with population_size
        """
        
        print(f"Initialize {self.population_size} organisms ...")
        population_params = self.random_parameters(self.lower_bound, self.upper_bound, (self.population_size, self.fitness_size))

        self.population = np.array([self.create_new_organism(p) for p in population_params], dtype=object)

        fitness_values = np.array([organism.fitness for organism in self.population])
        best_index = np.argmax(fitness_values)
        self.best_organism = self.population[best_index]
    
    def create_new_organism(self, attributes: np.ndarray) -> Organism:
        """
        Create a new organism with given attributes
        :param attributes: a list of attribute of organism [alpha, beta]
        :return: a new organism
        """
        
        new_organism = Organism(attributes)
        new_organism.aco_fitness()
        return new_organism
    
    def mutualism(self, i: int) -> None:
        """
        Execute mutualism on the current best organism to find a better organism \n
        Idea: Given an organism Xi, another different organism Xj ̸= Xi is chosen from the population. A mutualism operation is performed for Xi and
        Xj in order to enhance their survival ability in the ecosystem. The new candidates Xinew and Xjnew are created as:
                                        Xinew = Xi + rand(0, 1) × (Xbest − Mutual_Vector × BF1)\n
                                        Xjnew = Xj + rand(0, 1) × (Xbest − Mutual_Vector × BF2)\n
                                        Mutual_Vector = (Xi + Xj)/2\n
        Xinew and Xjnew are accepted only if their fitness values are higher than those of the Xi and Xj
        """
        
        indices = np.arange(self.population_size)
        mask = indices != i
        available_indices = indices[mask]
        j = np.random.choice(available_indices)

        # Pick two organisms Xi and Xj
        Xj = self.population[j]
        Xi = self.population[i]

        # Create random values for BF1, BF2
        bf1 = np.random.randint(1, 3)
        bf2 = np.random.randint(1, 3)
        array_rand = self.random_parameters(0, 1, self.fitness_size)

        # Calculate mutual vector
        mutual_vector = (Xi.phenotypes + Xj.phenotypes) / 2
        
        new_Xi_params = Xi.phenotypes + array_rand * (self.best_organism.phenotypes - mutual_vector * bf1)
        new_Xj_params = Xj.phenotypes + array_rand * (self.best_organism.phenotypes - mutual_vector * bf2)


        new_Xi_params = np.clip(new_Xi_params, self.lower_bound, self.upper_bound)
        new_Xj_params = np.clip(new_Xj_params, self.lower_bound, self.upper_bound)

        # Create new organisms
        new_Xi = self.create_new_organism(attributes=new_Xi_params)
        new_Xj = self.create_new_organism(attributes=new_Xj_params)
        self.population[i] = new_Xi if new_Xi.fitness > Xi.fitness else Xi
        self.population[j] = new_Xj if new_Xj.fitness > Xj.fitness else Xj
    
    def commensalism(self, i: int) -> None:
        """
        Execute commensalism on the current best organism to find a better organism \n
        Idea: Given an organism Xi, another organism Xj is selected at random from the population. Xi will be converted
        into a new organism under the help of Xj. Xinew will be accepted only if the fitness value is
        higher than that of the ancestor Xi. In the commensalism phase, the current best solution Xbest is taken as the reference organism
        for updating Xi. It aims to compute a promising organism near Xbest\n
                                Xinew = Xi + rand(−1, 1) × (Xbest − Xj)\n
        Xinew will be accepted only if the fitness value is higher than that of the ancestor Xi, It aims to compute a promising organism near
        Xbest
        """
        
        indices = np.arange(self.population_size)
        mask = indices != i
        available_indices = indices[mask]
        j = np.random.choice(available_indices)

        # Pick two organisms Xi and Xj
        Xj = self.population[j]
        Xi = self.population[i]

        array_rand = self.random_parameters(-1, 1, self.fitness_size)
        
        new_Xi_params = Xi.phenotypes + array_rand * (self.best_organism.phenotypes - Xj.phenotypes)
        
        new_Xi_params = np.clip(new_Xi_params, self.lower_bound, self.upper_bound)

        # Create new organism
        new_Xi = self.create_new_organism(attributes=new_Xi_params)
        self.population[i] = new_Xi if new_Xi.fitness > Xi.fitness else Xi
    
    def parasitism(self, i: int) -> None:
        """
        Execute parasitism on the current best organism to find a better organism\n
        Idea: an organism Xi is selected and copied as an artificial parasite called Parasite_Vector. Then, Parasite_Vector is modified in some dimension computed
        with a random number function. At last, an organism Xj is selected as a host for comparison. If the Parasite_Vector has a better
        fitness value, it will replace Xj in the population and Xj will be deleted. Otherwise, Xj is maintained and Parasite_Vector will be
        neglected.
        :param a_index: Position of an organism in the list of organisms
        """
        
        parasite_params = np.copy(self.population[i].phenotypes)

        # Pick a random organism from the population
        indices = np.arange(self.population_size)
        mask = indices != i
        available_indices = indices[mask]
        j = np.random.choice(available_indices)

        Xj = self.population[j]

        # Transform parasite with random dimension
        dim_index = np.random.randint(0, self.fitness_size)
        
        if isinstance(self.lower_bound, np.ndarray) and isinstance(self.upper_bound, np.ndarray):
            parasite_params[dim_index] = self.random_parameters(self.lower_bound[dim_index], self.upper_bound[dim_index])
        else:
            parasite_params[dim_index] = self.random_parameters(self.lower_bound, self.upper_bound)

        # Create parasite organism
        parasite = self.create_new_organism(parasite_params)
        self.population[j] = parasite if parasite.fitness > Xj.fitness else Xj
    
    def excute_sos(self) -> None:
        """
        Excute SOS algorithm
        """
        print("Executing SOS algorithm...")
        
        # fitness_values = np.array([organism.fitness for organism in self.population])
        # best_index = np.argmax(fitness_values)
        
        # self.mutualism(i=best_index)
        # self.commensalism(i=best_index)
        # self.parasitism(i=best_index)
        
        # # Update best organism
        # fitness_values = np.array([organism.fitness for organism in self.population])
        # best_index = np.argmax(fitness_values)
        # self.best_organism = self.population[best_index]
        
        for i, val in enumerate(self.population):
            self.mutualism(i)
            self.commensalism(i)
            self.parasitism(i)
                
            fitness_values = np.array([organism.fitness for organism in self.population])
            best_index = np.argmax(fitness_values)
            self.best_organism = self.population[best_index]
                
        print(f'Finished SOS algorithm')
    
    def set_best_organism(self) -> None:
        """
        Set the best organism from the population
        """
        
        fitness_values = np.array([organism.fitness for organism in self.population])
        best_index = np.argmax(fitness_values)
        self.best_organism = self.population[best_index]