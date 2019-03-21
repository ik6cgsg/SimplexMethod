import numpy
import pandas
from linprog.general import getDualTask, CompSign, Task
from simplex.simplex_table import initSimplex, simplex

# Output settings
desired_width = 600
pandas.set_option('display.width', desired_width)
numpy.set_printoptions(linewidth=desired_width)

# 23 transport task
haveProductColumn = [22, 19, 14, 15]
m = len(haveProductColumn)
needProductRow = [24, 13, 4, 17, 12]
n = len(needProductRow)
transportCostTable = [
    [ 3, 17,  2, 19,  6],
    [15,  8, 12,  5, 10],
    [ 8,  9,  8,  9, 14],
    [ 5,  7,  6,  3, 18]
]

def printMatr(matrix):
    print(numpy.matrix(matrix))

def getStandartForm(transportCostTable, haveProductColumn, needProductRow):
    matrix = numpy.zeros((m + n, n * m))
    targetFun = []
    restrictions = numpy.concatenate((haveProductColumn, needProductRow))
    #restrictions = numpy.concatenate((restrictions, numpy.zeros((n * m - n - m))))
    # http://data.cyclowiki.org/images/thumb/9/92/%D0%94%D0%B0%D0%BD%D1%86%D0%B8%D0%B3_296.png/800px-%D0%94%D0%B0%D0%BD%D1%86%D0%B8%D0%B3_296.png
    # http://cyclowiki.org/wiki/%D0%A0%D0%B5%D1%88%D0%B5%D0%BD%D0%B8%D0%B5_%D1%82%D1%80%D0%B0%D0%BD%D1%81%D0%BF%D0%BE%D1%80%D1%82%D0%BD%D0%BE%D0%B9_%D0%B7%D0%B0%D0%B4%D0%B0%D1%87%D0%B8_%D1%81%D0%B8%D0%BC%D0%BF%D0%BB%D0%B5%D0%BA%D1%81-%D0%BC%D0%B5%D1%82%D0%BE%D0%B4%D0%BE%D0%BC
    for i in range(m):
        for j in range(i * n, (i + 1) * n):
            matrix[i][j] = 1

    for i in range(m, m + n):
        for j in range(m):
            matrix[i][(i - m) + j * n] = 1

    for i in range(m):
        targetFun = numpy.concatenate((targetFun, transportCostTable[i]))

    return matrix, restrictions, targetFun

# EX1
matrixEx = [
    [2, 1, 1, 1, 3],
    [3, 0, 2, -1, 6],
    [1, 0, -1, 2, 1]
]
targetEx = [0, 0, 3, -2, -1]
restrictEx = [5, 7, 2]
compEx = [CompSign.EQUAL, CompSign.EQUAL, CompSign.EQUAL]
taskEx = Task.MAXIMIZE

# EX2
matrixEx2 = [
    [-1, 2, 3],
    [3, -1, 1],
    [-5, 4, 1]
]
targetEx2 = [12, 24, 18]
restrictEx2 = [2, 1, 3]
compEx2 = [CompSign.GREATER_EQUAL, CompSign.GREATER_EQUAL, CompSign.GREATER_EQUAL,
           CompSign.ANY, CompSign.ANY, CompSign.ANY]
taskEx2 = Task.MINIMIZE

# EX3
m = [
    [2, -1],
    [1, -5]
]
t = [2, -1]
r = [2, -4]
co = [CompSign.LESS_EQUAL, CompSign.LESS_EQUAL, CompSign.GREATER_EQUAL, CompSign.GREATER_EQUAL]
tsk = Task.MAXIMIZE

def testDual():
    m, r, t, c, ts = getDualTask(matrixEx2, restrictEx2, targetEx2, compEx2, taskEx2)
    print("Dual:")
    print(m, r, t, c, ts, sep="\n")

def testSimplex():
    N, B, A, b, c, v = initSimplex(m, r, t, co, tsk)
    print("Init:")
    print(N, B, A, b, c, v, sep="\n")
    #x = simplex(matrixEx, restrictEx, targetEx)
    #print("Solution:")
    #print(x)

def solveTask():
    # Getting form of linear programming task
    matrix, restrictions, target = getStandartForm(transportCostTable, haveProductColumn, needProductRow)
    print("Direct task:\n")
    printMatr(matrix)
    print("Target function: ", target)
    print("Restrictions: ", restrictions)
    # Go from minimize -> to maximize
    matrix, restrictions, target = getDualTask(matrix, restrictions, target)
    print("Dual task:\n")
    printMatr(matrix)
    print("Target function: ", target)
    print("Restrictions: ", restrictions)
    # Find solution by simplex method (Cormen)
    x = simplex(matrix, restrictions, target)
    print("Solution:")
    print(x)

def main():
    print("Hello, python!")
    # testDual()
    testSimplex()
    # solveTask()


if __name__ == '__main__':
    main()
