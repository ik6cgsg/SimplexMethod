import numpy
import pandas
from linprog.general import getDualTask, CompSign, Task
from simplex.simplex_cormen import initSimplex, simplex

# Output settings
desired_width = 600
pandas.set_option('display.width', desired_width)
numpy.set_printoptions(linewidth=desired_width)

# 23 transport task
haveProductColumn = [22, 19, 14, 15]
needProductRow = [24, 13, 4, 17, 12]
transportCostTable = [
    [ 3, 17,  2, 19,  6],
    [15,  8, 12,  5, 10],
    [ 8,  9,  8,  9, 14],
    [ 5,  7,  6,  3, 18]
]

def printMatr(matrix):
    print(numpy.matrix(matrix))

def getStandartForm(transportCostTable, haveProductColumn, needProductRow):
    m = len(haveProductColumn)
    n = len(needProductRow)
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

    comps = numpy.full(len(restrictions) + len(targetFun), CompSign.GREATER_EQUAL)
    for i in range(len(restrictions)):
        comps[i] = CompSign.EQUAL

    task = Task.MINIMIZE

    return matrix, restrictions, targetFun, comps, task

# Example
matrixEx = [
            [ 1, 0,  0, 1, -5],
            [ 0, 2,  1, 0,  2],
            [-6, 0, -3, 0,  1],
            [ 0, 3,  0, 2,  2],
            [ 6, 0,  7, 0, -5]
        ]
targetEx = [0, 1, 0, 4, 2]
freeEx = 0
restrictEx = [18, 1, 25, -11, 3]
compEx = [CompSign.EQUAL, CompSign.EQUAL, CompSign.GREATER_EQUAL, CompSign.EQUAL, CompSign.LESS_EQUAL,
          CompSign.ANY, CompSign.ANY,
          CompSign.ANY, CompSign.GREATER_EQUAL,
          CompSign.ANY]
taskEx = Task.MAXIMIZE

def testDual():
    m, r, t, c, ts = getDualTask(matrixEx, restrictEx, targetEx, compEx, taskEx)
    print("Dual:")
    print(m, r, t, c, ts, sep="\n")

def testSimplex():
    #N, B, A, b, c, v = initSimplex(matrixEx, restrictEx, targetEx, compEx, taskEx, freeEx)
    #print("Init:")
    #print(N, B, A, b, c, v, sep="\n")
    x, v = simplex(matrixEx, restrictEx, targetEx, compEx, taskEx, freeEx)
    print("Solution:")
    print(x)
    print("Value:")
    print(v)

def solveTask():
    # Getting form of linear programming task
    matrix, restrictions, target, comps, task = getStandartForm(transportCostTable, haveProductColumn, needProductRow)
    print("Direct task:\n")
    printMatr(matrix)
    print("Target function: ", target)
    print("Restrictions: ", restrictions)
    # Go from minimize -> to maximize
    '''
    matrix, restrictions, target = getDualTask(matrix, restrictions, target)
    print("Dual task:\n")
    printMatr(matrix)
    print("Target function: ", target)
    print("Restrictions: ", restrictions)
    '''
    # Find solution by simplex method (Cormen)
    x, val = simplex(matrix, restrictions, target, comps, task, 0)
    print("Solution:")
    print(x)
    print("Value:")
    print(val)

def main():
    print("Hello, python!")
    # testDual()
    testSimplex()
    # solveTask()


if __name__ == '__main__':
    main()
