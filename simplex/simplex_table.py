import numpy

N = 0
M = 0

def initSimplex(matrix, restrictions, targetFun):
    minRestr = numpy.amin(restrictions)
    N = len(targetFun)
    M = len(restrictions)
    if minRestr >= 0:
        return list(range(N)), list(range(N, N + M)), \
               matrix, restrictions, targetFun, 0
    #TODO: else


def pivot(nonBasisVars, basisVars, matrix, restrictions, targetFun, freeTerm, srcIndex, distIndex):
    restrictions[distIndex] = restrictions[srcIndex] / matrix[srcIndex][distIndex]

    return