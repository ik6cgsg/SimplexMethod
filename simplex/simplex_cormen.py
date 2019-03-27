import numpy
from linprog.general import Task, getDualTask, CompSign


def getSimplexForm(nonBasis, basis, matrix, restr, target):
    basis = numpy.array(basis, dtype=int)
    nonBasis = numpy.array(nonBasis, dtype=int)
    n = len(nonBasis)
    m = len(basis)
    size = n + m
    sMatr = numpy.zeros((size, size), dtype=float)
    # Building simplex form matrix
    for i in range(size):
        if i in nonBasis:
            sMatr[i][i] = 1
            continue
        for j in nonBasis:
            sMatr[i][j] = matrix[i - n][j]
    sRestr = numpy.concatenate((numpy.zeros(n), restr))
    sRestr = sRestr.astype(float)
    sTarget = numpy.concatenate((target, numpy.full(size - len(target), 0)))
    sTarget = sTarget.astype(float)
    return nonBasis, basis, sMatr, sRestr, sTarget

def getCanonicalForm(matrix, restrictions, targetFun, compSigns, task):
    if task == Task.MINIMIZE:
        matrix, restrictions, targetFun, compSigns, task = \
            getDualTask(matrix, restrictions, targetFun, compSigns, task)
    N = len(restrictions)
    for i in range(N, len(compSigns)):
        if compSigns[i] == CompSign.ANY:
            print("Shit")
            l = len(targetFun)
            matrix = numpy.insert(matrix, l, 0, 1)
            matrix = numpy.insert(matrix, l, 0, 1)
            for j in range(N):
                matrix[j][l] = matrix[j][i - N]
                matrix[j][l + 1] = matrix[j][i - N] * -1
                matrix[j][i - N] = 0
            targetFun = numpy.concatenate((targetFun, numpy.zeros(2)))
            targetFun[l] = targetFun[i - N]
            targetFun[l + 1] = targetFun[i - N] * -1
            targetFun[i - N] = 0

    for cs in compSigns:
        if cs == CompSign.EQUAL:
            N += 1
    for i in range(N):
        if compSigns[i] == CompSign.EQUAL:
            compSigns[i] = CompSign.LESS_EQUAL
            compSigns = numpy.insert(compSigns, i + 1, CompSign.LESS_EQUAL)
            restrictions = numpy.insert(restrictions, i + 1, -restrictions[i])
            matrix = numpy.insert(matrix, i + 1, numpy.array(matrix[i]) * -1, 0)
        elif compSigns[i] == CompSign.GREATER_EQUAL:
            compSigns[i] = CompSign.LESS_EQUAL
            restrictions[i] = -restrictions[i]
            matrix[i] = matrix[i] * -1

    print("----Canon form----")
    printDebugInfo(228, [1488], [322], matrix, restrictions, targetFun, 0)

    return matrix, restrictions, targetFun

def initSimplex(matrix, restrictions, targetFun, compSigns, task, free):
    matrix, restrictions, targetFun = getCanonicalForm(matrix, restrictions, targetFun, compSigns, task)
    minRestr = numpy.amin(restrictions)
    n = len(targetFun)
    m = len(restrictions)
    if minRestr >= 0:
        N, B, A, b, c = getSimplexForm(list(range(n)), list(range(n, n + m)), matrix, restrictions, targetFun)
        return N, B, A, b, c, free
    # Auxiliary system
    mAux = numpy.insert(matrix, 0, -1, 1)
    tAux = [-1]
    nonBasis, basis, mAux, restrictions, tAux = getSimplexForm(list(range(n + 1)), list(range(n + 1, n + 1 + m)),
                                                                     mAux, restrictions, tAux)
    minInds = numpy.where(restrictions == minRestr)
    k = minInds[0][0]
    l = k
    nonBasis, basis, mAux, restrictions, tAux, free = pivot(nonBasis, basis, mAux, restrictions, tAux, free, l, 0)
    x, nonBasis, basis, mAux, restrictions, tAux, free = simplexCycle(nonBasis, basis, mAux, restrictions, tAux, free)
    if x[0] == 0:
        if 0 in basis:
            nonBasis, basis, mAux, restrictions, tAux, free = pivot(nonBasis, basis, mAux, restrictions, tAux, free, 0, l)
        tAux[0] = 0
        for i in range(len(targetFun)):
            tAux[i + 1] = targetFun[i]
        for i in basis:
            free += tAux[i] * restrictions[i]
            for j in nonBasis:
                tAux[j] -= tAux[i] * mAux[i][j]
            tAux[i] = 0
        mAux = numpy.delete(mAux, 0, 1)
        mAux = numpy.delete(mAux, 0, 0)
        tAux = numpy.delete(tAux, 0)
        nonBasis = nonBasis[nonBasis != 0]
        basis = numpy.array([x - 1 for x in basis])
        nonBasis = numpy.array([x - 1 for x in nonBasis])
        restrictions = numpy.delete(restrictions, 0)
        return nonBasis, basis, mAux, restrictions, tAux, free
    else:
        print("No solution")
        return -1


def pivot(nonBasisInd, basisInd, matrix, restrictions, targetFun, freeTerm, srcIndex, dstIndex):
    # Computing coeffs of equation for new basis var x[distIndex]
    restrictions[dstIndex] = restrictions[srcIndex] / matrix[srcIndex][dstIndex]
    for i in nonBasisInd:
        if i == dstIndex:
            continue
        matrix[dstIndex][i] = matrix[srcIndex][i] / matrix[srcIndex][dstIndex]
    matrix[dstIndex][srcIndex] = 1 / matrix[srcIndex][dstIndex]
    matrix[dstIndex][dstIndex] = 0
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
        matrix[i][dstIndex] = 0
    # Computing target func
    freeTerm += targetFun[dstIndex] * restrictions[dstIndex]
    for j in nonBasisInd:
        if j == dstIndex:
            continue
        targetFun[j] -= targetFun[dstIndex] * matrix[dstIndex][j]
    targetFun[srcIndex] = -targetFun[dstIndex] * matrix[dstIndex][srcIndex]
    # Computing new non basis ans basis sets
    nonBasisInd = nonBasisInd[nonBasisInd != dstIndex]
    nonBasisInd = numpy.insert(nonBasisInd, 0, srcIndex)
    basisInd = basisInd[basisInd != srcIndex]
    basisInd = numpy.insert(basisInd, 0, dstIndex)
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
    print(A)
    print("Restrictions: ", b)
    print("Target function: ", c)
    print("Free term in target func (max of func): ", v)
    print()

def simplexCycle(nonBasisInd, basisInd, sMatrix, sRestr, sTarget, freeTerm):
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
    return solution, nonBasisInd, basisInd, sMatrix, sRestr, sTarget, freeTerm

def simplex(matrix, restrictions, targetFun, compSigns, task, free):
    try:
        nonBasisInd, basisInd, sMatrix, sRestr, sTarget, free = initSimplex(matrix, restrictions,
                                                                            targetFun, compSigns, task, free)
    except (ValueError, TypeError):
        return "no", "solution"
    solution, _, _, _, _, _, value = \
        simplexCycle(nonBasisInd, basisInd, sMatrix, sRestr, sTarget, free)
    return solution, value