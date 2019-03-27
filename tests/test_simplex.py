import pytest

from linprog.general import CompSign, Task
from simplex.simplex_cormen import simplex

treshold = 0.00001

def assertEqf(var1, var2):
    assert(var1 - treshold <= var2 <= var1 + treshold)

class TestSimplex:
    def test1(self):
        matrixEx = [
            [2, 1, 1, 1, 3],
            [3, 0, 2, -1, 6],
            [1, 0, -1, 2, 1]
        ]
        targetEx = [0, 0, 3, -2, -1]
        restrictEx = [5, 7, 2]
        freeEx = 0
        compEx = [CompSign.EQUAL, CompSign.EQUAL, CompSign.EQUAL,
                  CompSign.GREATER_EQUAL, CompSign.GREATER_EQUAL, CompSign.GREATER_EQUAL,
                  CompSign.GREATER_EQUAL, CompSign.GREATER_EQUAL]
        taskEx = Task.MAXIMIZE
        x, v = simplex(matrixEx, restrictEx, targetEx, compEx, taskEx, freeEx)
        l = len(targetEx)
        xExpect = [2, 0, 2/3, 1/3, 0]
        vExpect = 4/3
        for i in range(l):
            assertEqf(x[i], xExpect[i])
        assertEqf(v, vExpect)

    def test2(self):
        matrixEx = [
            [-2, 1, 1, 0, 0],
            [1, -2, 0, 2, 0],
            [1, 0, 0, 0, -1]
        ]
        targetEx = [-1, 1, 0, 0, 0]
        freeEx = 0
        restrictEx = [2, 2, 5]
        compEx = [CompSign.EQUAL, CompSign.EQUAL, CompSign.EQUAL,
                  CompSign.GREATER_EQUAL, CompSign.GREATER_EQUAL, CompSign.GREATER_EQUAL,
                  CompSign.GREATER_EQUAL, CompSign.GREATER_EQUAL]
        taskEx = Task.MINIMIZE
        x, v = simplex(matrixEx, restrictEx, targetEx, compEx, taskEx, freeEx)
        assert(x == "no")
        assert(v == "solution")

    def test3(self):
        matrixEx = [
            [-1, 1, -2, 2, 6],
            [-1, -2, 1, -7, -3],
            [1, -1, -3, 1, 0]
        ]
        targetEx = [-2, 1, -3, 2, 10]
        freeEx = 6
        restrictEx = [-2, -5, -4]
        compEx = [CompSign.EQUAL, CompSign.EQUAL, CompSign.EQUAL,
                  CompSign.GREATER_EQUAL, CompSign.GREATER_EQUAL, CompSign.GREATER_EQUAL,
                  CompSign.GREATER_EQUAL, CompSign.GREATER_EQUAL]
        taskEx = Task.MINIMIZE
        x, v = simplex(matrixEx, restrictEx, targetEx, compEx, taskEx, freeEx)
        vExpect = 1
        assertEqf(v, vExpect)

    def test4(self):
        matrixEx = [
            [2, 0, 1, -1, 1],
            [1, 0, -1, 2, 1],
            [0, 2, 1, -1, 2],
            [1, 0, 0, 1, -5]
        ]
        targetEx = [3, -2, 0, -5, 1]
        freeEx = 0
        restrictEx = [2, 3, 6, 8]
        compEx = [CompSign.LESS_EQUAL, CompSign.LESS_EQUAL, CompSign.LESS_EQUAL, CompSign.GREATER_EQUAL,
                  CompSign.GREATER_EQUAL, CompSign.GREATER_EQUAL, CompSign.GREATER_EQUAL,
                  CompSign.GREATER_EQUAL, CompSign.GREATER_EQUAL]
        taskEx = Task.MAXIMIZE
        x, v = simplex(matrixEx, restrictEx, targetEx, compEx, taskEx, freeEx)
        assert(x == "no")
        assert(v == "solution")

    def test5(self):
        matrixEx = [
            [1, -2, 3, 0, -5],
            [0, 2, 1, 6, 2],
            [6, 0, -3, 0, 1],
            [1, 3, 4, 5, 2],
            [6, 0, 7, -10, 5]
        ]
        targetEx = [13, -2, 11, 4, 2]
        freeEx = 0
        restrictEx = [10, 20, -8, 5, 15]
        compEx = [CompSign.EQUAL, CompSign.LESS_EQUAL, CompSign.GREATER_EQUAL, CompSign.EQUAL, CompSign.EQUAL,
                  CompSign.ANY, CompSign.ANY,
                  CompSign.ANY, CompSign.ANY,
                  CompSign.GREATER_EQUAL]
        taskEx = Task.MAXIMIZE
        x, v = simplex(matrixEx, restrictEx, targetEx, compEx, taskEx, freeEx)
        #assert(x == 4)
        assert(v == (1287 * 101 + 22) / 101 / 3)