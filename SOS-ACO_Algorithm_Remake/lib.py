import numpy as np
import numba
from scipy.spatial.distance import cdist

# Input: 2 arrays of city positions: point1 = [x1, y1], point2 = [x2, y2]
# Output: distance between two cities. Type of return value: float
# Function: calculate the Euclidean distance between 2 points.
# @numba.jit(nopython=True)
# def euclidean_distance(point1: np.ndarray, point2: np.ndarray):
#     return np.sqrt(np.sum((point1 - point2) ** 2))

# Input: two 2D arrays of city positions:
#   points1 = [[x0,y0], [x1,y1], ...], points2 = [[x0,y0], [x1,y1], ...]
# Output: distance matrix: A[m x n].
#   m: number of cities in points1
#   n: number of cities in points2
#   A[i,j]: distance from city[i] in points1 to city[j] in points2
# Function: calculate pairwise distances between cities.
# @numba.jit(nopython=True)
# def pairwise_distances(points1: np.ndarray, points2: np.ndarray):
#     m, n = points1.shape[0], points2.shape[0]
#     distances = np.empty((m, n), dtype=np.float64)
#     for i in range(m):
#         for j in range(n):
#             distances[i, j] = euclidean_distance(points1[i], points2[j])
#     return distances

def pairwise_distances(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return cdist(X, Y, metric='euclidean')

# Input: Matrix to round, afterDot: number of digits after decimal.
# Output: Rounded matrix.
# Function: Round each cell in the matrix.
# @numba.jit(nopython=True)
# def round_matrix(matrix: np.ndarray, afterDot: int):
#     return np.round(matrix, afterDot)

# Input: matrix: 2D NumPy array
# Output: matrix with inverse values for each cell
# Function: calculate the inverse (1/x) of each cell in 2D matrix.
# @numba.jit(nopython=True)
# def heuristic_matrix(matrix: np.ndarray):
#     m, n = matrix.shape
#     result = np.zeros_like(matrix)
#     for i in range(m):
#         for j in range(n):
#             if matrix[i, j] != 0:
#                 result[i, j] = 1.0 / matrix[i, j]
#     return result

def heuristic_matrix(distances: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    if np.any(distances < 0) or np.any(np.isnan(distances)):
        raise ValueError("Ma trận khoảng cách chứa giá trị không hợp lệ (âm hoặc NaN)")
    heuristics = np.where(distances > epsilon, 1.0 / np.maximum(distances, epsilon), 0)
    
    return heuristics

# Input: pheromone_matrix: 2D array, heuristic_matrix: 2D array, alpha: int, beta: int
# Output: res: 2D array
# Function: calculate probability for each cell using:
#   res[i,j] = (pheromone_matrix[i,j] ** alpha) * (heuristic_matrix[i,j] ** beta)
@numba.jit(nopython=True)
def probability_matrix(pheromone_matrix: np.ndarray, heuristic_matrix: np.ndarray, alpha: int, beta: int):
    return (pheromone_matrix ** alpha) * (heuristic_matrix ** beta)