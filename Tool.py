# coding:utf-8
import numpy as np


def mylistRound(arr):
    l = len(arr)
    for i in range(l):
        arr[i] = round(arr[i])
    return arr


def find_all_index(arr, item):
    return [i for i, a in enumerate(arr) if a == item]


def find_all_index_not(arr, item):
    l = len(arr)
    flag = np.zeros(l)
    index = find_all_index(arr, item)
    flag[index] = 1
    not_index = find_all_index(flag, 0)
    return not_index


def NDS(fit1, fit2):
    v = 0
    dom_less = 0;
    dom_equal = 0;
    dom_more = 0;
    for k in range(2):
        if fit1[k] > fit2[k]:
            dom_more = dom_more + 1
        elif fit1[k] == fit2[k]:
            dom_equal = dom_equal + 1
        else:
            dom_less = dom_less + 1
    if dom_less == 0 and dom_equal != 2:
        v = 2
    if dom_more == 0 and dom_equal != 2:
        v = 1
    return v

def NDS2(fit1, fit2, weight):
    v = 0
    dom_less = 0
    dom_equal = 0
    dom_more = 0

    if fit1[0] / weight[0] <= fit1[1] / weight[1]:
        TCHfitness1 = fit1[0] / weight[0]
    if fit2[0] / weight[0] < fit2[1] / weight[1]:
        TCHfitness2 = fit2[0] / weight[0]
    if TCHfitness1 > TCHfitness2:
        dom_more = dom_more + 1
    else:
        dom_less = dom_less + 1

    if TCHfitness1 > TCHfitness2:
        dom_more = dom_more + 1
    elif TCHfitness2 > TCHfitness1:
        dom_less = dom_less + 1
    else:
        dom_equal = dom_equal + 1
    if dom_less == 1:
        v = 2
    if dom_more == 1:
        v = 1
    return v

def Ismemeber(item, list):
    l = len(list)
    flag = 0
    for i in range(l):
        if list[i] == item:
            flag = 1
            break
    return flag


def DeleteReapt(QP, QM, QF, QFit, QFtime, QFinish, QMT, ps):
    row = np.size(QFit, 0)
    i = 0
    while i < row:
        if i >= row:
            break

        F = QFit[i, :]
        j = i + 1
        while j < row:
            if QFit[j][0] == F[0] and QFit[j][1] == F[1]:
                QP = np.delete(QP, j, axis=0)
                QM = np.delete(QM, j, axis=0)
                QF = np.delete(QF, j, axis=0)
                QFit = np.delete(QFit, j, axis=0)
                QFtime = np.delete(QFtime, j, axis=0)
                QFinish = np.delete(QFinish, j, axis=0)
                QMT = np.delete(QMT, j, axis=0)
                j = j - 1
                row = row - 1
                if row < 2 * ps + 1:
                    break
            j = j + 1
        i = i + 1
        if row < 2 * ps + 1:
            break
    return QP, QM, QF, QFit, QFtime, QFinish, QMT


def DeleteReaptE(QP, QM, QF, QFit, QFtime, QFinish, QMT, Qual):  # for elite strategy
    row = np.size(QFit, 0)
    i = 0
    while i < row:
        if i >= row:
            # print('break 1')
            break

        F = QFit[i, :]
        j = i + 1
        while j < row:
            if QFit[j][0] == F[0] and QFit[j][1] == F[1]:
                QP = np.delete(QP, j, axis=0)
                QM = np.delete(QM, j, axis=0)
                QF = np.delete(QF, j, axis=0)
                QFit = np.delete(QFit, j, axis=0)
                QFtime = np.delete(QFtime, j, axis=0)
                QFinish = np.delete(QFinish, j, axis=0)
                QMT = np.delete(QMT, j, axis=0)
                Qual = np.delete(Qual, j, axis=0)
                j = j - 1
                row = row - 1
            j = j + 1
        i = i + 1

    return QP, QM, QF, QFit, QFtime, QFinish, QMT, Qual


def pareto(fitness):
    PF = []
    L = np.size(fitness, axis=0)
    pn = np.zeros(L, dtype=int)
    for i in range(L):
        for j in range(L):
            dom_less = 0;
            dom_equal = 0;
            dom_more = 0;
            for k in range(2):  # number of objectives
                if (fitness[i][k] > fitness[j][k]):
                    dom_more = dom_more + 1
                elif (fitness[i][k] == fitness[j][k]):
                    dom_equal = dom_equal + 1
                else:
                    dom_less = dom_less + 1

            if dom_less == 0 and dom_equal != 2:  # i is dominated by j
                pn[i] = pn[i] + 1;
        if pn[i] == 0:  # add i into pareto front
            PF.append(i)
    return PF

def ndsort(fitness):
    N, M = fitness.shape
    front_no = np.full(N, np.inf)
    max_fno = 0

    rank = np.argsort(fitness[:, 0])
    sort_fitness = fitness[rank]

    while np.sum(front_no < np.inf) < N:
        max_fno += 1
        for i in range(N):
            if front_no[i] == np.inf:
                dominated = False
                for j in range(i - 1, -1, -1):
                    if front_no[j] == max_fno:
                        if sort_fitness[i, 1] <= sort_fitness[j, 1]:
                            dominated = True
                            break
                if not dominated:
                    front_no[i] = max_fno

    front_no[rank] = front_no

    front_numbers = np.zeros(max_fno)
    for i in range(1, max_fno + 1):
        front_numbers[i - 1] = np.sum(front_no == i)

    return front_numbers, front_no, max_fno