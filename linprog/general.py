import numpy


def getDualTask(matrix, restrictions, targetFun):
    dualMatrix = numpy.transpose(matrix)
    dualTargetFun = restrictions
    dualRestrictions = targetFun

    return dualMatrix, dualRestrictions, dualTargetFun
