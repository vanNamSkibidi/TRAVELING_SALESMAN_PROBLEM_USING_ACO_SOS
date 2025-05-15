# import numpy as np
from ACO import *
from parameter import *
import random
import lib

class Organism(object):
    def __init__(self, phenotypes : List[int]) -> None:
        """
        Organism in Symbiotic optimization search

        :param phenotypes: a list of attribute values of organism [alpha, beta]
        """
        self.phenotypes = phenotypes  # phenotype
        self.fitness = 0  # value of the fitness function
        self.ACO = None

    def setFitness(self, cost : float) -> None:
        """
        Calculate fitness value of organism
        
        :param cost: the length of best path
        """
        self.fitness = 1 / cost

    def aco_fitness(self) -> None:
        """
        Use set of parameters in organism object to find path for TSP via ACO
        """
        towns = TOWNS
        self.ACO = ACO(ants=ANTS, evaporation_rate=EVAPORATION_RATE, intensification=INTENSIFICATION, alpha=self.phenotypes[0], beta=self.phenotypes[1], beta_evaporation_rate=0.005)
        self.ACO.fit(towns, iterations=int(1 * MAX_ITER_ACO), conv_crit=25, verbose = False)
        self.setFitness(self.ACO.best)

    def __str__(self):
        return '{0} = {1}'.format(self.phenotypes, self.fitness)


class SOS(object):
    def __init__(self,
                 l_bound : float,
                 u_bound : float,
                 population_size : int,
                 fitness_vector_size : int,
                 ants : int) -> None:
        
        """
        Symbiotic optimization search. Finds optimal set of parameters for Ant colony optimization.

        :param l_bound: lower bound of range value limit
        :param u_bound: upper bound of range value limit
        :param population_size: number of organisms in population
        :param fitness_vector_size: number of parameters contained in an organism
        :param ants: number of ants to traverse the graph
        """

        self.l_bound = l_bound
        self.u_bound = u_bound
        self.population_size = population_size
        self.fitness_vector_size = fitness_vector_size
        self.population = None
        self.best = None
        self.best_idx = None
        self.ants = ants

    def float_rand(self, a:float, b:float, size = None) -> List[float]:
        """
        Generate a list have size = size parameter of numbers between a and b
        :param a: a number (int, float)
        :param b: a number (int, float)
        :return: a list have size = size parameter of numbers between a and b
        """
        if size == None:
            return a + ((b - a) * random.random())
        res = []
        if type(size) == int :
            for _ in range(size):
                res.append(a + (b - a) * random.random())
            return res
        for i in range (size[0]):
            temp = []
            for j in range(size[1]):
                temp.append(a + (b - a) * random.random())
            res.append(temp)
        return res
    
    def generate_population(self) -> None:
        """
        Initializa a population of population_size organisms
        """
        print(f"Initialize {self.population_size} organisms ...")
        population = self.float_rand(self.l_bound, self.u_bound, [self.population_size, self.fitness_vector_size])
        self.population = [self.create_new_organism(p) for p in population]
        self.best = sorted(self.population, key=lambda x: x.fitness, reverse=True)[0] 

    def create_new_organism(self, attributes : List[float]) -> Organism:
        """
        Create new organism with alpha, beta values in
        :param attributes: list of value attributes of new organism
        """
        new_organism = Organism(attributes)
        new_organism.aco_fitness()
        return new_organism

    def mutualism(self, a_index : int) -> None:
        """
        Execute mutualism on the current best organism to find a better organism \n
        Idea: Given an organism Xi, another different organism Xj ̸= Xi is chosen from the population. A mutualism operation is performed for Xi and
        Xj in order to enhance their survival ability in the ecosystem. The new candidates Xinew and Xjnew are created as:
                                        Xinew = Xi + rand(0, 1) × (Xbest − Mutual_Vector × BF1)\n
                                        Xjnew = Xj + rand(0, 1) × (Xbest − Mutual_Vector × BF2)\n
                                        Mutual_Vector = (Xi + Xj)/2\n
        Xinew and Xjnew are accepted only if their fitness values are higher than those of the Xi and Xj

        :param a_index: Position of an organism in the list of organisms
        """
        population = lib.arange(0, self.population_size)
        population.remove(a_index)

        random.shuffle(population)
        b_index = population[0]

        b = self.population[b_index]
        a = self.population[a_index]
        bf1 = random.randint(1, 3)
        bf2 = random.randint(1, 3)

        array_rand = self.float_rand(0, 1, self.fitness_vector_size)

        mutual = lib.divideMatrix(lib.addMatrix(a.phenotypes,  b.phenotypes),  2)
        new_a = lib.addMatrix(a.phenotypes, lib.multiMatrix(lib.minusMatrix(self.best.phenotypes, mutual * bf1), array_rand))
        new_b = lib.addMatrix(b.phenotypes, lib.multiMatrix(lib.minusMatrix(self.best.phenotypes, mutual * bf2), array_rand))
        new_a = self.create_new_organism([self.u_bound if x > self.u_bound
                            else self.l_bound if x < self.l_bound else x for x in new_a])
        new_b = self.create_new_organism(attributes=[self.u_bound if x > self.u_bound
                            else self.l_bound if x < self.l_bound else x for x in new_b])

        self.population[a_index] = new_a if new_a.fitness > a.fitness else a
        self.population[b_index] = new_b if new_b.fitness > b.fitness else b

    def commensalism(self, a_index) -> None:
        """
        Execute commensalism on the current best organism to find a better organism \n
        Idea: Given an organism Xi, another organism Xj is selected at random from the population. Xi will be converted
        into a new organism under the help of Xj. Xinew will be accepted only if the fitness value is
        higher than that of the ancestor Xi. In the commensalism phase, the current best solution Xbest is taken as the reference organism
        for updating Xi. It aims to compute a promising organism near Xbest\n
                                Xinew = Xi + rand(−1, 1) × (Xbest − Xj)\n
        Xinew will be accepted only if the fitness value is higher than that of the ancestor Xi, It aims to compute a promising organism near
        Xbest

        :param a_index: Position of an organism in the list of organisms                    
        """

        population = lib.arange(0, self.population_size)
        population.remove(a_index)
        random.shuffle(population)
        b_index = population[0]

        b = self.population[b_index]
        a = self.population[a_index]

        array_rand = self.float_rand(-1, 1, self.fitness_vector_size)
        new_a = lib.addMatrix(a.phenotypes, lib.multiMatrix(lib.minusMatrix(self.best.phenotypes, b.phenotypes) , array_rand))
        new_a = self.create_new_organism(attributes=[self.u_bound if x > self.u_bound
                                else self.l_bound if x < self.l_bound
                                else x for x in new_a])
        self.population[a_index] = new_a if new_a.fitness > a.fitness else a


    def parasitism(self, a_index) -> None:
        """
        Execute parasitism on the current best organism to find a better organism\n
        Idea: an organism Xi is selected and copied as an artificial parasite called Parasite_Vector. Then, Parasite_Vector is modified in some dimension computed
        with a random number function. At last, an organism Xj is selected as a host for comparison. If the Parasite_Vector has a better
        fitness value, it will replace Xj in the population and Xj will be deleted. Otherwise, Xj is maintained and Parasite_Vector will be
        neglected.

        :param a_index: Position of an organism in the list of organisms
        """
        parasite = self.population[a_index].phenotypes

        population = lib.arange(0, self.population_size)
        population.remove(a_index)
        random.shuffle(population)
        b_index = population[0]

        b = self.population[b_index]

        parasite[random.randint(0, self.fitness_vector_size - 1)] = self.float_rand(self.l_bound, self.u_bound)
        parasite = self.create_new_organism(parasite)
        self.population[b_index] = parasite if parasite.fitness > b.fitness else b

    def sos_exe(self) -> None:
        """
        Execute mutualism, commensalism, parasitism on the current best organism to find a better organism
        """
        organism_idx = self.population.index(self.best)
        self.mutualism(a_index=organism_idx)
        self.commensalism(a_index=organism_idx)
        self.parasitism(a_index=organism_idx)
        self.best = sorted(self.population, key=lambda x: x.fitness, reverse=True)[0] # Get the largest

    def set_best_organism(self) -> None:
        """
        Set the best organism in SOS
        """
        self.best = sorted(self.population, key=lambda x: x.fitness, reverse=True)[0] # Get the largest