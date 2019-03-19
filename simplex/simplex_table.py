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
    sTarget = numpy.concatenate((target, numpy.zeros(m)))
    return nonBasis, basis, sMatr, sRestr, sTarget, 0

def initSimplex(matrix, restrictions, targetFun):
    minRestr = numpy.amin(restrictions)
    n = len(targetFun)
    m = len(restrictions)
    if minRestr >= 0:
        return getSimplexForm(list(range(n)), list(range(n, n + m)), matrix, restrictions, targetFun)
    #TODO: else

def pivot(nonBasisInd, basisInd, matrix, restrictions, targetFun, freeTerm, srcIndex, dstIndex):
    # Computing coeffs of equation for new basis var x[distIndex]
    restrictions[dstIndex] = restrictions[srcIndex] / matrix[srcIndex][dstIndex]
    for i in nonBasisInd:
        if i == dstIndex:
            continue
        matrix[dstIndex][i] = matrix[srcIndex][i] / matrix[srcIndex][dstIndex]
    matrix[dstIndex][srcIndex] = 1 / matrix[srcIndex][dstIndex]
    # Computing other equations coeffs
    for i in basisInd:
        if i == srcIndex:
            continue
        restrictions[i] -= matrix[i][dstIndex] * restrictions[dstIndex]
        for j in nonBasisInd:
            if j == dstIndex:
                continue
            matrix[i][j] -= matrix[i][dstIndex] * matrix[dstIndex][j]
        matrix[i][srcIndex] = -matrix[i][dstIndex] * matrix[dstIndex][srcIndex]
    # Computing target func
    freeTerm += targetFun[dstIndex] * restrictions[dstIndex]
    for j in nonBasisInd:
        if j == dstIndex:
            continue
        targetFun[j] -= targetFun[dstIndex] * matrix[dstIndex][j]
    targetFun[srcIndex] = -targetFun[dstIndex] * matrix[dstIndex][srcIndex]
    # Computing new non basis ans basis sets
    nonBasisInd.remove(dstIndex)
    nonBasisInd.append(srcIndex)
    basisInd.remove(srcIndex)
    basisInd.append(dstIndex)
    # Reset new non-basis var
    for j in range(len(targetFun)):
        if j == srcIndex:
            matrix[j][j] = 1
            continue
        matrix[srcIndex][j] = 0
    restrictions[srcIndex] = 0
    targetFun[dstIndex] = 0
    return nonBasisInd, basisInd, matrix, restrictions, targetFun, freeTerm

def consistPositive(targetFun):
    for coeff in targetFun:
        if coeff > 0:
            return True
    return False

def getFirstPositive(targetFun):
    for i in range(len(targetFun)):
        if targetFun[i] > 0:
            return i

def printDebugInfo(iter, N, B, A, b, c, v):
    print("Iteration: ", iter)
    print("Non basis indices: ", N)
    print("Basis indices: ", B)
    print("Matrix: ")
    print(numpy.matrix(A))
    print("Restrictions: ", b)
    print("Target function: ", c)
    print("Free term in target func (max of func): ", v)
    print()

def simplex(matrix, restrictions, targetFun):
    nonBasisInd, basisInd, sMatrix, sRestr, sTarget, freeTerm = initSimplex(matrix, restrictions, targetFun)
    iter = 0

    while consistPositive(sTarget):
        delta = numpy.full(len(basisInd) + len(nonBasisInd), numpy.inf)
        dstInd = getFirstPositive(sTarget)
        for i in basisInd:
            if sMatrix[i][dstInd] > 0:
                delta[i] = sRestr[i] / sMatrix[i][dstInd]

        srcInds = numpy.where(delta == numpy.amin(delta))
        srcInd = srcInds[0][0]
        if srcInd == numpy.inf:
            return "No solution\n"

        nonBasisInd, basisInd, sMatrix, sRestr, sTarget, freeTerm = pivot(nonBasisInd, basisInd, sMatrix, sRestr,
                                                                          sTarget, freeTerm, srcInd, dstInd)
        # DEBUG
        printDebugInfo(iter, nonBasisInd, basisInd, sMatrix, sRestr, sTarget, freeTerm)
        iter += 1

    solution = numpy.zeros(len(nonBasisInd))
    for i in range(len(nonBasisInd)):
        if i in basisInd:
            solution[i] = sRestr[i]
    return solution