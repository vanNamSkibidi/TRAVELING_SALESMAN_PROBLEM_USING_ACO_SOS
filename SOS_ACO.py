import time
from SOS import *
from Local_Optimize import *
import lib
import random

class SOS_ACO:
    def __init__(self, ants:int, evaporation_rate:float, intensification:float,
                 SOS_obj:SOS, beta_evaporation_rate:float) -> None:
        """
        Ant colony optimizer. Finds the path that either minimizes distance traveled between nodes. 
        This optimizer is devoted to solving Traveling Salesman Problem (TSP).

        :param ants: Number of ants to traverse the graph.
        :param evaporation_rate: Rate at which pheromone evaporates.
        :param intensification: Constant value added to the best path.
        :param alpha: Weight of pheromone.
        :param beta: Weight of heuristic (1/distance).
        :param beta_evaporation_rate: Rate at which beta decays. 
        :param choose_best: Probability to choose the best path.
        """

        # Parameters
        self.ants = ants
        self.evaporation_rate = evaporation_rate
        self.Q = intensification
        self.beta_evaporation_rate = beta_evaporation_rate
        self.SOS_obj = SOS_obj

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
        """ Get information of ACO in string format"""
        string = "Ant Colony Optimizer"
        string += "\n------------------------"
        string += "\nDesigned to solve travelling salesman problem. Optimizes either the minimum or maximum distance travelled."
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
        self.list_of_nodes = lib.arange(0, self.num_nodes)
        self.distance_matrix = lib.pairwise_distances(spatial_map, spatial_map)
        self.distance_matrix = lib.round_matrix(self.distance_matrix, 0)
        self.heuristic_matrix = lib.heuristic_matrix(self.distance_matrix)
        self.SOS_obj.generate_population()

    def __reset_list_of_nodes(self) -> None:
        """
        Reset the list of all nodes for the next iteration.
        """
        self.list_of_nodes = lib.arange(0, self.num_nodes)

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

    def __intensify(self, best_path: List[int], best : float) -> None:
        """
        Adds pheromone to the best path of organism in SOS.

        :param best_path: new best path obtained so far
        """
        for i in range(len(best_path) - 1):
            # self.pheromone_matrix[best_path[i], best_path[i+1]] += self.Q / len(best_path)
            self.pheromone_matrix[best_path[i]][best_path[i+1]] += self.Q / best

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

    def __update_probabilities(self) -> None:
        """
        Each iteration probability matrix needs to be updated. This function does exactly that.
        """

        self.probability_matrix = lib.probability_matrix(self.pheromone_matrix, self.heuristic_matrix, 
                                                        self.heuristic_alpha, self.heuristic_beta)

    def sos_optimize_best(self) -> None:
        """
        Execute SOS for searching new better alpha, beta parameters
        """
        self.SOS_obj.sos_exe()

        if self.pheromone_matrix is None:
            self.pheromone_matrix = self.SOS_obj.best.ACO.pheromone_matrix

        if self.best is None or self.best > self.SOS_obj.best.ACO.best:
            print("\n------------------- Update alpha, beta ---------------------------\n")
            self.best = self.SOS_obj.best.ACO.best
            self.best_path = self.SOS_obj.best.ACO.best_path
            self.__intensify(self.best_path, self.best)


        self.heuristic_alpha = self.SOS_obj.best.phenotypes[0]
        self.heuristic_beta = self.SOS_obj.best.phenotypes[1]
        self.pheromone_matrix = self.SOS_obj.best.ACO.pheromone_matrix

        self.__update_probabilities()


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
        self.__initialize(spatial_map=list(spatial_map))
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
            self.sos_optimize_best()
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

                # reset path 
                path = [None] * (self.num_nodes + 1)

            best_path, best_score = self.__evaluate(iteration=iteration, paths=paths)

            #  LOCAL OPTIMIZATION
            best_path, best_score, isBetter = local_optimize(best_hc=best_path, best_score=best_score, distance_matrix=self.distance_matrix)
            if isBetter :
                self.__intensify(best_path, best_score)


            if iteration == 0:
                best_score_so_far = best_score
                self.best_path = best_path

            if best_score < best_score_so_far:
                best_score_so_far = best_score
                self.best_path = best_path
                self.best = best_score_so_far
                num_equal = 0
                
            
                
            if best_score == best_score_so_far:
                num_equal += 1
            else:
                num_equal = 0

            self.best_series[iteration] = best_score
            self.global_best[iteration] = best_score_so_far

            self.__evaporate()
            self.__update_pheromone(iteration=iteration, paths=paths)

            print(f"======> Current alpha: {self.heuristic_alpha}")
            print(f"======> Current beta: {self.heuristic_beta}")

            if verbose:
                print(
                f"Iteration {iteration}/{iterations} | Score: {round(best_score, decimal)} | Best so far: {round(best_score_so_far, decimal)} | {round(time.time() - start_iter, decimal)} s | num_equal: {num_equal} |"
                    )
                
            if (best_score == best_score_so_far) and (num_equal == conv_crit):
                self.converged = True ; self.stopped_at_iteration = iteration
                if verbose: print("\nConvergence criterion has been met. Stopping....")
                break

            self.SOS_obj.best.ACO.fitness = 1 / best_score
            self.SOS_obj.set_best_organism()


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




