from typing import List, Tuple

def local_optimize(best_hc:List[int], best_score:float, distance_matrix:List[List[float]]) -> Tuple[List[int], float]:
    """
    This local optimization strategy is used to improve the solution quality searched by SOS–ACO and accelerate
    the convergence rate.
    Idea: Given a TSP instance of size n, an HC is denoted as HC = (v1, v2, . . . , vn), where vi is one of the n cities and i represents
    the position of city vi in HC. There are two steps for improving an HC using the local optimization strategy.\n
            Step 1: For each city vk in an HC, find the city vk' which is nearest to vk in the following cities. As shown in Eq. (9), dis(vi
    , vj) is the distance between vi and vj
            Step 2: Reverse the path between vk and vk' so that vk is adjacent to vk'. If the adjusted path is shorter than the original path, the adjusted path is preserved. Otherwise, the original path
    is maintained.

    :param best_hc: The path is evaluated as the best path by Ant Colony Optimization
    :param best_score: The score of best_hc
    :param distance_matrix: The matrix of distance costs between each pairs of cities
    """

    best_hc = list(best_hc[:-1])
    n = len(best_hc)
    for k in range(n-2):    
        k_ = k + 1
        d_min = distance_matrix[best_hc[k]][best_hc[k_]]

        for i in range(k+2, n):
            d = distance_matrix[best_hc[k]][best_hc[i]]
            if d < d_min:
                k_ = i
        if k_ != k + 1:
            reverse_hc = reverse_oper(solution=best_hc, k=k, k_=k_)
            best_score_r = cal_score_hc(new_hc=reverse_hc, distance_matrix=distance_matrix)
            if best_score_r < best_score:
                reverse_hc.append(reverse_hc[0])
                return reverse_hc, best_score_r, True
    best_hc.append(best_hc[0])
    return best_hc, best_score, False


def cal_score_hc(new_hc:List[int], distance_matrix:List[List[float]]) -> float:
    """
    Calculate the score of new path base on distance cost in distance_matrix

    :param new_hc: The new path obtained by reverse operation
    :param distance_matrix: The matrix of distance costs between each pairs of cities
    """
    dis = 0
    for i in range(len(new_hc)-1):
        dis += distance_matrix[new_hc[i]][new_hc[i+1]]
    dis += distance_matrix[new_hc[i+1]][new_hc[0]]
    return dis

def reverse_oper(solution:List[int], k:int, k_:int) -> List[int]:
    """
    Reverse position of cities such that city_k is adjacent to city k_
    :param solution: The new path obtained by reverse operation
    :param k: The position of city_k
    :param k_: The position of city_k_
    """
    reverse = solution.copy()
    sub_cur = reverse[k:k_]
    sub_cur.reverse()
    reverse[k:k_] = sub_cur
    return reverse

