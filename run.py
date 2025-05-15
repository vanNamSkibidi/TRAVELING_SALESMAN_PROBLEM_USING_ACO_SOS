import os
from SOS import *
from parameter import *
from SOS_ACO import *

def main():
    """
    - Change parameter of the algorithm in parameter.py file (e.g., PATH_TO_MAP to specify the .tsp file)
    - Read readme.txt for detail guidance
    - Saves results for the current TOWNS into a file in the ACO_SOS_FED folder
    """
    
    output_folder = "ACO_SOS_FED"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Extract the base name of the .tsp file (e.g., "eil51" from "eil51.tsp")
    tsp_basename = os.path.splitext(os.path.basename(PATH_TO_MAP))[0]
    output_file_path = os.path.join(output_folder, f"{tsp_basename}.txt")

    # Open the output file for writing
    with open(output_file_path, "w") as output_file:
        towns = TOWNS
        if ANTS < 2:
            message = "Need at least two ants to run this algorithm\n"
            print(message)
            output_file.write(message)
            return

        sos = SOS(l_bound=PARAMETER_SPACE[0], u_bound=PARAMETER_SPACE[1],
                  population_size=POP_SIZE,
                  fitness_vector_size=DIM,
                  ants=ANTS)

        ACO_optimizer = SOS_ACO(ants=ANTS, evaporation_rate=EVAPORATION_RATE,
                                intensification=INTENSIFICATION, SOS_obj=sos,
                                beta_evaporation_rate=BETA_EVAPORATION_RATE)
        ACO_optimizer.fit(spatial_map=towns, iterations=int(MAX_ITER_ACO), conv_crit=25)
        best_path, best, fit_time, best_series = ACO_optimizer.get_result()
        best_path_coor = [towns[i] for i in best_path]

        # Write results to the file and print to terminal
        result = f"# Results for TOWNS from {tsp_basename}.tsp\n"
        result += f"Best path: {best_path_coor}\n"
        result += f"Best distance: {best}\n"
        result += f"Fit time: {fit_time} seconds\n"
        result += f"Best series: {best_series}\n"

        output_file.write(result)
        print(result)

if __name__ == '__main__':
    main()