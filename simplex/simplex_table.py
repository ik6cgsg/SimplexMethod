import numpy

def getSimplexForm(nonBasis, basis, matrix, restr, target):
    n = len(nonBasis)
    m = len(basis)
    size = n + m
    sMatr = numpy.zeros((size, size))
    # Building simplex form matrix
    for i in range(size):
        if i in nonBasis:
            sMatr[i][i] = 1
            continue
        for j in nonBasis:
            sMatr[i][j] = matrix[i - n][j]
    sRestr = numpy.concatenate((numpy.zeros(n), restr))
    sTarget = numpy.concatenate((target, numpy.zeros(n)))
    return nonBasis, basis, sMatr, sRestr, sTarget, 0

def initSimplex(matrix, restrictions, targetFun):
    minRestr = numpy.amin(restrictions)
    n = len(targetFun)
    m = len(restrictions)
    if minRestr >= 0:
        return getSimplexForm(list(range(n)), list(range(n, n + m)), matrix, restrictions, targetFun)
    #TODO: else

def pivot(nonBasisInd, basisInd, matrix, restrictions, targetFun, freeTerm, srcIndex, distIndex):
    # Computing coeffs of equation for new basis var x[distIndex]
    restrictions[distIndex] = restrictions[srcIndex] / matrix[srcIndex][distIndex]
    for i in nonBasisInd:
        if i == distIndex:
            continue
        matrix[distIndex][i] = matrix[srcIndex][i] / matrix[srcIndex][distIndex]
    matrix[distIndex][srcIndex] = 1 / matrix[srcIndex][distIndex]
    # Computing other equations coeffs
    for i in basisInd:
        if i == srcIndex:
            continue
        restrictions[i] -= matrix[i][distIndex] * restrictions[distIndex]
        for j in nonBasisInd:
            if j == distIndex:
                continue
            matrix[i][j] -= matrix[i][distIndex] * matrix[distIndex][j]
        matrix[i][srcIndex] = -matrix[i][distIndex] * matrix[distIndex][srcIndex]
    # Computing target func
    freeTerm += targetFun[distIndex] * restrictions[distIndex]
    for j in nonBasisInd:
        if j == distIndex:
            continue
        targetFun[j] -= targetFun[distIndex] * matrix[distIndex][j]
    targetFun[srcIndex] = -targetFun[distIndex] * matrix[distIndex][srcIndex]
    # Computing new non basis ans basis sets
    nonBasisInd.remove(distIndex)
    nonBasisInd.append(srcIndex)
    basisInd.remove(srcIndex)
    basisInd.append(distIndex)
    # Reset new non-basis var
    for j in range(len(targetFun)):
        if j == srcIndex:
            matrix[j][j] = 1
            continue
        matrix[srcIndex][j] = 0
    restrictions[srcIndex] = 0
    targetFun[distIndex] = 0
    return nonBasisInd, basisInd, matrix, restrictions, targetFun, freeTerm

def simplex(matrix, restrictions, targetFun):
    nonBasisInd, BasisInd, sMatrix, sRestr, sTarget, freeTerm = initSimplex(matrix, restrictions, targetFun)
    