from typing import List
def read_map(file_path:str) -> List[List[float]]:
    """
    Read all point coordianates from data file

    :param file_path: path to tsp file
    """
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


PATH_TO_MAP = "./Benchmark/rd400.tsp"
TOWNS = read_map(PATH_TO_MAP)

# Parameter for SOS
PARAMETER_SPACE = (1, 5)
POP_SIZE = 10  # The number of candidate solutions
DIM = 2  # number of problem variables [α, β]

# Parameter for ACO
ANTS = 20
EVAPORATION_RATE = 0.1
INTENSIFICATION = 1000
MAX_ITER_ACO = 50  # The number of iterations
BETA_EVAPORATION_RATE = 0.05