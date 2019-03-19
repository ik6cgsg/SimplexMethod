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

def getDualTask(matrix, restrictions, targetFun, compSigns, task):
    dualMatrix = numpy.transpose(matrix)
    dualTargetFun = restrictions
    dualRestrictions = targetFun
    compLen = len(compSigns)
    targLen = len(targetFun)
    dualCompSigns = numpy.zeros(compLen)
    # min -> max and vice versa
    if task == Task.MAXIMIZE:
        dualTask = Task.MINIMIZE
        # equations
        for i in range(targLen):
            if compSigns[i] == CompSign.LESS_EQUAL:
                dualCompSigns[i - targLen] = CompSign.GREATER_EQUAL
            elif compSigns[i] == CompSign.EQUAL:
                dualCompSigns[i - targLen] = CompSign.ANY
            else:
                print("Cant solve this(")
                exit(-1)
        # variables
        for i in range(targLen, compLen):
            if compSigns[i] == CompSign.GREATER_EQUAL:
                dualCompSigns[i + targLen] = CompSign.GREATER_EQUAL
            elif compSigns[i] == CompSign.ANY:
                dualCompSigns[i + targLen] = CompSign.EQUAL
            else:
                print("Cant solve this(")
                exit(-1)
    else:
        dualTask = Task.MAXIMIZE
        # equations
        for i in range(targLen):
            if compSigns[i] == CompSign.GREATER_EQUAL:
                dualCompSigns[i - targLen] = CompSign.GREATER_EQUAL
            elif compSigns[i] == CompSign.EQUAL:
                dualCompSigns[i - targLen] = CompSign.ANY
            else:
                print("Cant solve this(")
                exit(-1)
        # variables
        for i in range(targLen, compLen):
            if compSigns[i] == CompSign.GREATER_EQUAL:
                dualCompSigns[i + targLen] = CompSign.LESS_EQUAL
            elif compSigns[i] == CompSign.ANY:
                dualCompSigns[i + targLen] = CompSign.EQUAL
            else:
                print("Cant solve this(")
                exit(-1)
    return dualMatrix, dualRestrictions, dualTargetFun, compSigns, dualTask
