from Local_Optimize import *
import time
from parameter import *
import lib
import random
from typing import Tuple, List

class ACO:
    def __init__(self, ants : int, evaporation_rate : float, intensification : float,
                 alpha : float, beta : float, beta_evaporation_rate : float) -> None:
            
        """
        Ant colony optimizer. Finds the path that either maximizes or minimizes distance traveled between nodes. 
        This optimizer is devoted to solving Traveling Salesman Problem (TSP).

        :param ants: Number of ants to traverse the graph.
        :param evaporation_rate: Rate at which pheromone evaporates.
        :param intensification: Constant value added to the best path.
        :param alpha: Weight of pheromone.
        :param beta: Weight of heuristic (1/distance).
        :param beta_evaporation_rate: Rate at which beta decays.
        """

        # Parameters
        self.ants = ants
        self.evaporation_rate = evaporation_rate
        self.Q = intensification
        self.heuristic_alpha = alpha
        self.heuristic_beta = beta
        self.beta_evaporation_rate = beta_evaporation_rate

        # Matrices
        self.distance_matrix = None
        self.pheromone_matrix = None
        self.heuristic_matrix = None
        self.probability_matrix = None

        self.spatial_map = None
        self.num_nodes = None
        self.list_of_nodes = None

        # Internal statistics
        self.global_best = None
        self.best_series = None
        self.best_path = None
        self.best = None

        # Optimizer's status
        self.fitted = False
        self.fit_time = None

        # Early stopping 
        self.converged = False
        self.stopped_at_iteration = None

    def __str__(self) -> str:
        """ Get information of ACO_LO in string format """

        string = "Ant Colony Optimizer"
        string += "\n------------------------"
        string += "\nDesigned to solve traveling salesman problem. Optimizes either the minimum or maximum distance traveled."
        string += "\n------------------------"
        string += f"\nNumber of ants:\t\t\t\t{self.ants}"
        string += f"\nEvaporation rate:\t\t\t{self.evaporation_rate}"
        string += f"\nIntensification factor:\t\t\t{self.Q}"
        string += f"\nAlpha Heuristic:\t\t\t{self.heuristic_alpha}"
        string += f"\nBeta Heuristic:\t\t\t\t{self.heuristic_beta}"
        string += f"\nBeta Evaporation Rate:\t\t\t{self.beta_evaporation_rate}"
        string += "\n------------------------"

        if self.fitted:
            string += "\n\nThis optimizer has been fitted."
        else:
            string += "\n\nThis optimizer has NOT been fitted."
        return string
    
    def __initialize(self, spatial_map: List[List[float]]) -> None:
        """
        Initializes various matrices and checks given list of points.
        
        :param spatial_map: list of coordinates of cities in 2d plane. Ex: [[x1,y1],[x2,y2]...[xb, yn]]
        """
        for _ in spatial_map:
            assert len(_) == 2, "These are not valid points! Maybe check them? :)"

        self.spatial_map = spatial_map
        self.num_nodes = len(self.spatial_map)

        self.distance_matrix = lib.pairwise_distances(spatial_map, spatial_map)


        self.pheromone_matrix = [[1 for _ in range(self.num_nodes)] for _ in range(self.num_nodes)]

        self.heuristic_matrix = lib.heuristic_matrix(self.distance_matrix)

        lib.fill_diagonal(self.pheromone_matrix, val=0)
        lib.fill_diagonal(self.heuristic_matrix, val=0)

        
        self.probability_matrix = lib.probability_matrix(self.pheromone_matrix, self.heuristic_matrix, 
                                                         self.heuristic_alpha, self.heuristic_beta)

        self.list_of_nodes = lib.arange(0, self.num_nodes)



    def __reset_list_of_nodes(self) -> None:
        """
        Reset the list of all nodes for the next iteration.
        """
        self.list_of_nodes = lib.arange(0, self.num_nodes)

    def __update_probabilities(self) -> None:
        """
        Each iteration probability matrix needs to be updated. This function does exactly that.
        """

        self.probability_matrix = lib.probability_matrix(self.pheromone_matrix, self.heuristic_matrix, 
                                                         self.heuristic_alpha, self.heuristic_beta)


    def __travel_to_next_node_from(self, current_node: int) -> int:
        """
        Chooses the next node based on probabilities. 

        :param current_node: The node an ant is currently at.
        """

        numerator = []
        for i in range(len(self.probability_matrix[current_node])):
            if i in self.list_of_nodes:
                numerator.append(self.probability_matrix[current_node][i])
        sum_allowed_k = sum(numerator)
        probabilities = [x / sum_allowed_k for x in numerator]
        next_node = random.choices(self.list_of_nodes, probabilities)[0]
        return next_node

    def __remove_node(self, node: int) -> None:
        """
        Removes the node after an ant has traveled through it.
        :param node: The node an ant is currently at.
        """

        index = 0
        for i in range(len(self.list_of_nodes)):
            if self.list_of_nodes[i] == node :
                index = i
                break

        self.list_of_nodes[index] = -1

        self.list_of_nodes = [x for x in self.list_of_nodes if x != -1]

    def __evaluate(self, iteration: int, paths: List) -> Tuple[List[int], float]:
        """
        Evaluates ants' solution from iteration. Given all paths that ants have chosen, pick the best one.

        :param iteration: Iteration's number.
        :param paths: Ants' paths form iteration.
        :return: Best path and best score.
        """

        paths = paths[iteration]
        scores = [0] * len(paths)
        
        for i, path in enumerate(paths):
            score = 0
            for index in range(len(path) - 1):
                score += self.distance_matrix[path[index]][path[index+1]]
            scores[i] = score

        best = min(enumerate(scores), key=lambda x: x[1])[0]

        return paths[best], scores[best]
    
  
    def __evaporate(self) -> None:
        """
        Evaporates pheromone every iteration. Also evaporates beta parameter (optional).
        """

        for i in range (len(self.pheromone_matrix)):
            for j in range (len(self.pheromone_matrix[i])):
                self.pheromone_matrix[i][j] *= (1 - self.evaporation_rate)

        self.heuristic_beta *= (1 - self.beta_evaporation_rate)

    def __update_pheromone(self, iteration : int, paths : List[List[int]]) -> None:
        """
        Update pheromone on path the ants pass.
        
        :param iteration: Iteration's number.
        :param paths: Ants' paths form iteration.
        """

        paths_iter = paths[iteration]
        for path_kth in paths_iter:
            L_k = 0
            for index in range(len(path_kth) - 1):
                L_k += self.distance_matrix[path_kth[index]][path_kth[index+1]]
            for i in range(len(path_kth) - 1):
                self.pheromone_matrix[path_kth[i]][path_kth[i+1]] += self.Q / L_k # (∆τkij(t))

    def fit(self, spatial_map: List[List[float]], iterations : int, conv_crit = 20, verbose = True, decimal = 2) -> None:
        """
        The core function of the optimizer. It fits ACO to a specific map.

        :param spatial_map: List of positions (x, y) of the nodes.
        :param iterations: Number of iterations.
        :param conv_crit: Convergence criterion. The function stops after that many repeated scores. 
        :param verbose: If enabled ACO informs you about the progress.
        :param decimal: Number of decimal places. 
        """

        start = time.time()
        self.__initialize(spatial_map=spatial_map)

        num_equal = 0

        if verbose: print(f"{self.num_nodes} nodes were given. Beginning ACO Optimization with {iterations} iterations...\n")

        self.best_series = [0.0] * iterations
        self.global_best = [0.0] * iterations

        paths = []
        for _ in range(iterations):
            temp = []
            for _ in range(self.ants):
                temp1 = [None] * (self.num_nodes + 1)
                temp.append(temp1)
            paths.append(temp)

        path = [None] * (self.num_nodes + 1)
        for iteration in range(iterations):
            start_iter = time.time()
            for ant in range(self.ants):
                current_node = self.list_of_nodes[random.randint(0, self.num_nodes - 1)]
                start_node = current_node
                node = 0
                while True:
                    path[node] = current_node
                    
                    self.__remove_node(node=current_node)

                    if len(self.list_of_nodes) != 0:
                        current_node = self.__travel_to_next_node_from(current_node=current_node)
                        node += 1
                    else:
                        break 

                path[node + 1] = start_node # go back to start 
                self.__reset_list_of_nodes() 
                paths[iteration][ant] = path

                path = [None] * (self.num_nodes + 1)

            best_path, best_score = self.__evaluate(iteration=iteration, paths=paths)

            if iteration == 0:
                best_score_so_far = best_score
                self.best_path = best_path
            else:
                if best_score < best_score_so_far:
                    best_score_so_far = best_score
                    self.best_path = best_path

            if best_score == best_score_so_far:
                num_equal += 1
            else:
                num_equal = 0

            self.best_series[iteration] = best_score
            self.global_best[iteration] = best_score_so_far
            self.__evaporate()
            self.__update_pheromone(iteration=iteration, paths=paths)
            self.__update_probabilities()

            if verbose:
                print(
                    f"Iteration {iteration}/{iterations} | Score: {round(best_score, decimal)} | Best so far: {round(best_score_so_far, decimal)} | {round(time.time() - start_iter, decimal)} s"
                )

            if (best_score == best_score_so_far) and (num_equal == conv_crit):
                self.converged = True ; self.stopped_at_iteration = iteration
                if verbose: print("\nConvergence criterion has been met. Stopping....")
                break

        if not self.converged: self.stopped_at_iteration = iterations
        self.fit_time = round(time.time() - start)
        self.fitted = True

        if self.converged:
            temp = [x for x in self.best_series if x > 0]
            self.best_series = temp
        else:
            pass
        self.best = self.best_series[min(enumerate(self.best_series), key=lambda x: x[1])[0]]

        if verbose: print(
            f"\nACO fitted. Runtime: {self.fit_time // 60} minute(s). Best score: {round(self.best, decimal)}"
        )

    def get_result(self) -> Tuple[List[int], float, float, List[float]]:
        """
        :return: Tuple consisted of the best path, best distance, fit time, and list of each generation's best distance.
        """
        return self.best_path, self.best, self.fit_time, self.best_series

if __name__ == "__main__":
    towns = TOWNS
    ACO_optimizer = ACO(ants=ANTS, evaporation_rate=EVAPORATION_RATE, intensification=INTENSIFICATION, alpha=1.00, beta=5.00, beta_evaporation_rate=0.005)
    ACO_optimizer.fit(towns, iterations=MAX_ITER_ACO, conv_crit=25)
    best_path, best , fit_time , best_series = ACO_optimizer.get_result()

    best_path_coor = [towns[i] for i in best_path]
    print("Best path : ", best_path_coor)