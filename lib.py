import math

# Input : begin : int , end : int, step : int.
# Output: list.
# Function : create a list of integer number from begin to end - 1 with the distance between every number is step.
def arange(begin : int, end : int, step = 1 ):
        return [x for x in range(begin, end, step)]



    

# Input    : 2 list of position of city1 and city 2 : point1 = [x1, y1],  point2 = [x2, y2]
# Output   : distance between two city. Type of return value : double
# Function : calculate the distances between 2 cities.
def euclidean_distance(point1 : list, point2 : list):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(point1, point2)))


# Input    : two 2D - matrix of position of city: 
#   points1 = [[x0,y0], [x1,y1], [x2,y2], ...], points2 = [[x0,y0], [x1,y1], [x2,y2], ...]
# Output   : matrix distance of cities : A[m x n].
#   m : number of cities in points1
#   n : number of cities in points2
#   A[i][j] : the distance from city[i] in points1 to city[j] in point[2] 
# Function : calculate the distances between cities.
def pairwise_distances(points1 : list, points2 : list):
    distances = []
    for p1 in points1:
        row = []
        for p2 in points2:
            row.append(euclidean_distance(p1, p2))
        distances.append(row)
    return distances

# Input : Matrix need to round.  afterDot : number of digits after the dot.
# Output : Matrix has been rounded.
# Function : Round the value of each cell in matrix.
def round_matrix(matrix : list, afterDot : int):
    return [[round(x, afterDot) for x in row] for row in matrix]



# Input :  matrix[[,],[,] ,[,] , ...] : list - 2D matrix
# Output: matrix[[,],[,] ,[,] , ...] : list - the inverse values of each cell
# Function : calculate the inverse values of each cell in 2D matrix.
def heuristic_matrix(matrix : list):
    res = []
    for i in range(len(matrix)):
        res1 = []
        for j in range(len(matrix[i])):
            curr = matrix[i][j]
            if curr == 0 :
                res1.append(curr)
                continue
            
            curr = 1.0 / float(curr)
            res1.append(curr)
        res.append(res1)

    return res

# Input  : pheromone_matrix : 2D - list , heuristic_matrix : 2D -list, alpha : int , beta : int
# Output : res :2D list
# Function : calculate the probability each cell of res by equation :
#   res[i][j] = (pheromone_matrix[i][j] ** alpha ) * (heuristic_matrix[i][j] ** beta)
def probability_matrix(pheromone_matrix : list , heuristic_matrix : list, alpha : int , beta : int):
    res = []

    for i in range(len(heuristic_matrix)):
        res1 = []
        for j  in range(len(heuristic_matrix[i])):
            curr = (pheromone_matrix[i][j] ** alpha ) * (heuristic_matrix[i][j] ** beta)
            res1.append(curr)
        res.append(res1)

    return res

# Input : matrix[n x n] - 2D square matrix
# Output: matrix[n x n] with cell at main diagonal = val
# Function : fill cells in main diagonal with value val.
def fill_diagonal(matrix : list , val : int):
    for i in range(len(matrix)):
        matrix[i][i] = val

# Input : matrix1 : list , matrix2 : list  , 2 matrix need to be same dimension
# Output : res : list - sum of 2 matrix.
# Function : add two matrix.
def addMatrix(matrix1 : list, matrix2 : list):

    res = []


    if(type(matrix1[0]) != list):
        for i in range (len(matrix1)):
            res.append(matrix1[i] + matrix2[i])
        
        return res



    for i in range(len(matrix1)):
        temp = []
        for j in range(len(matrix1[i])):
            temp.append(matrix1[i][j] + matrix2[i][j])
        res.append(temp)

    return res 

# Input : matrix1 : list , matrix2 : list  , 2 matrix need to be same dimension
# Output : res : list - minus of 2 matrix.
# Function : minus two matrix.
def minusMatrix(matrix1 : list, matrix2 : list):
    res = []

    if(type(matrix1[0]) != list):
        for i in range (len(matrix1)):
            res.append(matrix1[i] - matrix2[i])
        
        return res


    for i in range(len(matrix1)):
        temp = []
        for j in range(len(matrix1[i])):
            temp.append(matrix1[i][j] - matrix2[i][j])
        res.append(temp)

    return res 


# Input : matrix1 : list , number which matrix need to divide
# Output : res : list
# Function : divide matrix by one number
def divideMatrix(matrix1 : list , number : float):
    res = []

    if(type(matrix1[0]) != list):
        for i in range (len(matrix1)):
            res.append(matrix1[i] / float(number))
        return res

    for i in range(len(matrix1)):
        temp = []
        for j in range(len(matrix1[i])):
            temp.append(matrix1[i][j] / float(number))
        res.append(temp)

    return res 
# Input : matrix1 : list , matrix2 : list, 2 matrix need to be same dimension
# Output : res : list
# Function : multiple A[i][j] with B[i][j]
def multiMatrix(matrix1 : list , matrix2 : list):
    res = []

    if(type(matrix1[0]) != list):
        for i in range (len(matrix1)):
            res.append(matrix1[i] * matrix2[i])
        return res

    for i in range(len(matrix1)):
        temp = []
        for j in range(len(matrix1[i])):
            temp.append(matrix1[i][j] * matrix2[i][j])
        res.append(temp)

    return res 





