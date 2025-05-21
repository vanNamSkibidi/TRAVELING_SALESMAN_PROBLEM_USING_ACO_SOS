from SOS import *
from parameter import *
from SOS_ACO import *
import os

def main():
    towns = TOWNS
    if ANTS < 2:
        print('Need at least two ants for this algorithm')
        return

    # Extract dataset name from PATH_TO_MAP (e.g., "eil51" from "Benchmark/eil51.tsp")
    data_name = os.path.basename(PATH_TO_MAP).split('.')[0]  # Gets "eil51"

    # Create output directory if it doesn't exist
    output_dir = "F:/2_2025/DAA/SOS-ACO_Algorithm_Remake/ACO_SOS_RFED"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Run the algorithm 20 times
    for run in range(1, 21):  # Runs from 1 to 20
        print(f"\nRun {run}/20 for dataset {data_name}...\n")
        
        # Initialize SOS and SOS_ACO
        sos = SOS(
            lower_bound=PARAMETER_SPACE[0],
            upper_bound=PARAMETER_SPACE[1],
            population_size=POP_SIZE,
            fitness_size=DIM,
            ants=ANTS
        )
        
        optimizer = SOS_ACO(
            ants=ANTS,
            evaporation_rate=EVAPORATION_RATE,
            intensification=INTENSIFICATION,
            SOS_obj=sos,
            beta_evaporation_rate=BETA_EVAPORATION_RATE
        )
        
        # Fit the model
        optimizer.fit(map_coordinates=towns, iterations=MAX_ITER_ACO, conv_crit=20)
        
        # Get results
        best_path, best, fit_time, best_series = optimizer.get_result()
        best_path_coordinates = [tuple(towns[i]) for i in best_path]
        
        # Prepare output content
        output = []
        output.append(f"Dataset: {data_name}")
        output.append(f"Run: {run}")
        output.append(f"Best path: {best_path_coordinates}")
        output.append(f"Best path length: {best}")
        output.append(f"Fitting time: {fit_time} seconds")
        output.append(f"Best series: {best_series.tolist()}")
        
        # Define output file path
        output_file = os.path.join(output_dir, f"{data_name}_iter{run}.txt")
        
        # Write to file
        with open(output_file, "w") as f:
            f.write("\n".join(map(str, output)))
        
        # Print to console as well
        print(f"\nResults saved to {output_file}")
        print("\n".join(map(str, output)))

if __name__ == "__main__":
    main()