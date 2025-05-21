import numpy as np
import time
from SOS import *
from Local_Optimize import *
import lib

class SOS_ACO:
    def __init__(self, ants: int, evaporation_rate: float, intensification: float,
                 SOS_obj: SOS, beta_evaporation_rate: float) -> None:
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
        
        self.ants = ants
        self.evaporation_rate = evaporation_rate
        self.Q = intensification
        self.beta_evaporation_rate = beta_evaporation_rate
        self.SOS_obj = SOS_obj
        
        self.distances = None
        self.pheromones = None
        self.heuristics = None
        self.probabilities = None
        
        self.map_coordinates = None
        self.num_nodes = None
        self.not_visited_nodes = None
        
        self.global_best = None
        self.best_series = None
        self.best_path = None
        self.best = None # Best cost over iterations
        
        self.fitted = False
        self.fitting_time = None
        
        self.converged = False
        self.stopped_at_iteration = None
    
    def __str__(self) -> str:
        string = "Ant Colony Optimizer"
        string += "\n------------------------"
        string += "\nDesigned to solve TSP. Optimizes either the minimum or maximum distance travelled."
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
    
    def __init_information(self, map_coordinates: np.ndarray) -> None:
        for i in map_coordinates:
            assert len(i) == 2, "Have some not valid points"
        
        self.map_coordinates = map_coordinates
        self.num_nodes = len(self.map_coordinates)
        
        self.distances = lib.pairwise_distances(self.map_coordinates, self.map_coordinates)
        self.distances = np.round(self.distances, 0)
        
        self.heuristics = lib.heuristic_matrix(self.distances)
        self.not_visited_nodes = np.arange(self.num_nodes)
        self.SOS_obj.gen_population()
    
    def __reset_nodes(self) -> None:
        """
        Reset all nodes for next iteration.
        """
        self.not_visited_nodes = np.arange(0, self.num_nodes)
    
    def __travel_next_node_from(self, current_node: int) -> int:
        """
        Travel to next node based on probabilities.
        """
        
        allowed_probs = self.probabilities[current_node, self.not_visited_nodes]
        
        # Normalize probabilities
        probabilities = allowed_probs / np.sum(allowed_probs)
        
        next_node = np.random.choice(self.not_visited_nodes, p=probabilities)
        
        return next_node
    
    def __remove_node(self, node: int) -> None:
        """
        Remove node after an ant has visited it.
        """
        
        index = np.where(self.not_visited_nodes == node)[0][0]
        self.not_visited_nodes = np.delete(self.not_visited_nodes, index)
    
    def __evaluate_solution(self, iteration: int, paths: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Evaluate the ant's solution of the current iteration.
        """
        paths_current = paths[iteration]
        
        scores = np.zeros(self.ants, dtype=float)
    
        for i, path in enumerate(paths_current):
            src_nodes = path[:-1]
            destination_nodes = path[1:]
            distances = self.distances[src_nodes, destination_nodes]
            scores[i] = np.sum(distances)
        
        best_index = np.argmin(scores)
        best_path = paths_current[best_index]
        best_score = scores[best_index]
        
        return best_path, best_score
    
    def __evaporate_pheromone(self) -> None:
        self.pheromones = self.pheromones * (1 - self.evaporation_rate)
    
    def __intensity(self, best_path: np.ndarray, best_cost: float) -> None:
        
        delta_pheromone = self.Q / best_cost
        
        from_nodes = best_path[:-1]
        to_nodes = best_path[1:]
        
        self.pheromones[from_nodes, to_nodes] += delta_pheromone
    
    def __update_pheromone(self, iteration: int, paths: np.ndarray) -> None:
        """
        Update pheromone on path after ants pass
        """
        paths_current = paths[iteration]
        L = np.zeros(paths_current.shape[0], dtype=float)
        for k, path in enumerate(paths_current):
            from_nodes = path[:-1]
            to_nodes = path[1:]
            distances = self.distances[from_nodes, to_nodes]
            L[k] = np.sum(distances)
        
        for k, path in enumerate(paths_current):
            from_nodes = path[:-1]
            to_nodes = path[1:]
            
            # Lượng pheromone được thêm: Q / L_k
            delta_pheromone = self.Q / L[k]
            
            # Cập nhật pheromone trên các cạnh của đường đi
            self.pheromones[from_nodes, to_nodes] += delta_pheromone
    
    def __update_probabilities(self) -> None:
        """
        Update probabilities matrix each iteration.
        """
        
        self.probabilities = lib.probability_matrix(self.pheromones, self.heuristics,
                                                    self.heuristic_alpha, self.heuristic_beta)
    
    def sos_optimize(self) -> None:
        """
        Optimize the parameters using SOS.
        """
        self.SOS_obj.excute_sos()

        # Ensure best_organism.ACO is initialized
        if self.SOS_obj.best_organism.ACO is None:
            self.SOS_obj.best_organism.compute_fitness(self.map_coordinates)

        # Update pheromones if not initialized
        if self.pheromones is None:
            self.pheromones = np.ones((self.num_nodes, self.num_nodes), dtype=np.float64)
            np.fill_diagonal(self.pheromones, 0)

        # Update best solution if improved
        if self.best is None or self.SOS_obj.best_organism.ACO.best < self.best:
            print("\n------------------- Update alpha, beta ---------------------------\n")
            self.best = self.SOS_obj.best_organism.ACO.best
            self.best_path = self.SOS_obj.best_organism.ACO.best_path
            self.__intensity(self.best_path, self.best)

        # Update alpha and beta
        self.heuristic_alpha = self.SOS_obj.best_organism.phenotypes[0]
        self.heuristic_beta = self.SOS_obj.best_organism.phenotypes[1]

        # Update probabilities
        self.__update_probabilities()
    
    def fit(self, map_coordinates: np.ndarray, iterations: int, conv_crit=20, verbose=True) -> None:
        """
        Fit the SOS-ACO model to the given map coordinates.

        :param map_coordinates: Coordinates of the cities.
        :param max_iter: Maximum number of iterations.
        """
        
        start = time.time()
        self.__init_information(map_coordinates=map_coordinates)
        
        nums_not_change = 0
        
        if verbose:
            print(f"{self.num_nodes} nodes were given. Beginning SOS-ACO Optimization with {iterations} iterations...\n")
            
        self.best_series = np.zeros(iterations, dtype=float)
        self.global_best = np.zeros(iterations, dtype=float)
        
        paths = np.full((iterations, self.ants, self.num_nodes + 1), -1, dtype=int)
        
        path = np.full(self.num_nodes + 1, -1, dtype=int)
        for iteration in range(iterations):
            self.sos_optimize()
            start_iteration = time.time()
            for ant in range(self.ants):
                current_node = self.not_visited_nodes[np.random.randint(0, self.num_nodes - 1)]
                start_node = current_node
                node_index = 0
                while True:
                    path[node_index] = current_node
                    
                    self.__remove_node(current_node)
                    
                    if self.not_visited_nodes.shape[0] != 0:
                        current_node = self.__travel_next_node_from(current_node=current_node)
                        node_index += 1
                    else:
                        break
                
                path[node_index + 1] = start_node # Add first node to complete HC
                self.__reset_nodes()
                paths[iteration, ant] = path
            
            best_path, best_score = self.__evaluate_solution(iteration=iteration, paths=paths)
            
            # Local optimization
            best_path, best_score, is_better = local_optimize(best_HC=best_path, best_cost=best_score, distances=self.distances)
            if is_better:
                self.__intensity(best_path, best_score)
            
            if iteration == 0:
                best_score_so_far = best_score
                self.best_path = best_path
            
            if best_score < best_score_so_far:
                best_score_so_far = best_score
                self.best_path = best_path
                self.best = best_score
                nums_not_change = 0
            
            if best_score == best_score_so_far:
                nums_not_change += 1
            else:
                nums_not_change = 0
            
            self.best_series[iteration] = best_score
            self.global_best[iteration] = best_score_so_far
            
            self.__evaporate_pheromone()
            self.__update_pheromone(iteration=iteration,paths=paths)
            
            print(f'================== Current alpha: {self.heuristic_alpha} ==================')
            print(f'================== Current beta: {self.heuristic_beta} ==================')
            
            if verbose:
                print(
                f"Iteration {iteration}/{iterations} | Score: {round(best_score, 3)} | Best so far: {round(best_score_so_far, 3)} | {round(time.time() - start_iteration, 3)} s")
            if best_score == best_score_so_far and nums_not_change == conv_crit:
                self.converged = True
                self.stop_iteration = iteration
                print('\nConvergence criterion has been met. Stop')
                break
            
            self.SOS_obj.best_organism.fitness = 1 / best_score
            self.SOS_obj.set_best_organism()
        
        end = time.time()
         
        if not self.converged:
            self.stopped_at_iteration = iterations
        self.fitting_time = round(end - start)
        self.fitted = True
        
        if self.converged:
            self.best_series = self.best_series[self.best_series > 0]
        self.best = float(np.min(self.best_series))
        
        if verbose:
            print(f'\nSOS-ACO fitted.\tRuntime: {self.fitting_time} seconds.\tBest cost: {round(self.best, 3)}')
    
    def get_result(self) -> Tuple[np.ndarray, float, float, np.ndarray]:
        return self.best_path, self.best, self.fitting_time, self.best_series
            
            