from enum import Enum
import numpy

class CompSign(Enum):
    LESS_EQUAL = -1
    EQUAL = 0
    GREATER_EQUAL = 1
    ANY = 2

class Task(Enum):
    MINIMIZE = 0
    MAXIMIZE = 1

# compSigns - for all equations + variables
def getDualTask(matrix, restrictions, targetFun, compSigns, task):
    dualMatrix = numpy.transpose(matrix)
    dualTargetFun = restrictions
    dualRestrictions = targetFun
    compLen = len(compSigns)
    targLen = len(targetFun)
    rLen = len(restrictions)
    dualCompSigns = numpy.full(compLen, CompSign.GREATER_EQUAL)
    # min -> max and vice versa
    if task == Task.MAXIMIZE:
        dualTask = Task.MINIMIZE
        # equations
        for i in range(rLen):
            if compSigns[i] == CompSign.LESS_EQUAL:
                dualCompSigns[i + rLen] = CompSign.GREATER_EQUAL
            elif compSigns[i] == CompSign.EQUAL:
                dualCompSigns[i + rLen] = CompSign.ANY
            else:
                dualCompSigns[i + rLen] = CompSign.GREATER_EQUAL
                dualTargetFun[i] = dualTargetFun[i] * -1
                for j in range(targLen):
                    dualMatrix[j][i] = dualMatrix[j][i] * -1
                #print("Cant solve this(")
                #exit(-1)
        # variables
        for i in range(rLen, compLen):
            if compSigns[i] == CompSign.GREATER_EQUAL:
                dualCompSigns[i - rLen] = CompSign.GREATER_EQUAL
            elif compSigns[i] == CompSign.ANY:
                dualCompSigns[i - rLen] = CompSign.EQUAL
            else:
                print("Cant solve this(")
                exit(-1)
    else:
        dualTask = Task.MAXIMIZE
        # equations
        for i in range(rLen):
            if compSigns[i] == CompSign.GREATER_EQUAL:
                dualCompSigns[i + rLen] = CompSign.GREATER_EQUAL
            elif compSigns[i] == CompSign.EQUAL:
                dualCompSigns[i + rLen] = CompSign.ANY
            else:
                dualCompSigns[i + rLen] = CompSign.GREATER_EQUAL
                dualTargetFun[i] = dualTargetFun[i] * -1
                for j in range(targLen):
                    dualMatrix[j][i] = dualMatrix[j][i] * -1
                #print("Cant solve this(")
                #exit(-1)
        # variables
        for i in range(rLen, compLen):
            if compSigns[i] == CompSign.GREATER_EQUAL:
                dualCompSigns[i - rLen] = CompSign.LESS_EQUAL
            elif compSigns[i] == CompSign.ANY:
                dualCompSigns[i - rLen] = CompSign.EQUAL
            else:
                print("Cant solve this(")
                exit(-1)
    return dualMatrix, dualRestrictions, dualTargetFun, dualCompSigns, dualTask
