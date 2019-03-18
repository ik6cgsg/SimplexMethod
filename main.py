import numpy

from linprog.general import getDualTask
from simplex.simplex_table import initSimplex, pivot

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

matrixEx = [
    [1, 1, 3],
    [2, 2, 5],
    [4, 1, 2]
]
targetEx = [3, 1, 2]
restrictEx = [30, 24, 36]

def main():
    """
    matrix, restrictions, target = getStandartForm(transportCostTable, haveProductColumn, needProductRow)
    print("Direct task:\n")
    printMatr(matrix)
    print("Target function: ", target)
    print("Restrictions: ", restrictions)

    matrix, restrictions, target = getDualTask(matrix, restrictions, target)
    print("Dual task:\n")
    printMatr(matrix)
    print("Target function: ", target)
    print("Restrictions: ", restrictions)
    """
    N, B, A, b, c, v = initSimplex(matrixEx, restrictEx, targetEx)
    print("Init:")
    print(N, B, A, b, c, v, sep="\n")
    print("Pivot 1 to 6:")
    N, B, A, b, c, v = pivot(N, B, A, b, c, v, 5, 0)
    print(N, B, A, b, c, v, sep="\n")


if __name__ == '__main__':
    main()
