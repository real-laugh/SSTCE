# coding:utf-8
import copy
import random
import Tool
import numpy as np
import math
from Tool import *

def crossover_OSMS(P1, M1, F1, P2, M2, F2, N, SH):
    NP1 = P1
    NM1 = M1
    NP2 = P2
    NM2 = M2
    NF1 = F1
    NF2 = F2
    # index of each operation in P1 and P2
    ci1 = np.zeros(SH, dtype=int)
    ci2 = np.zeros(SH, dtype=int)
    temp = [random.random() for _ in range(N)]
    temp = Tool.mylistRound(temp)
    J1 = find_all_index(temp, 1)

    for j in range(SH):
        if Ismemeber(P1[j], J1) == 1:
            ci1[j] = P1[j] + 1
        if Ismemeber(P2[j], J1) == 0:
            ci2[j] = P2[j] + 1
    index_1_1 = find_all_index(ci1, 0)
    index_1_2 = find_all_index_not(ci2, 0)

    index_2_1 = find_all_index(ci2, 0)
    index_2_2 = find_all_index_not(ci1, 0)
    l1 = len(index_1_1)
    l2 = len(index_2_1)
    # ①
    for j in range(l1):
        ci1[index_1_1[j]] = NP2[index_1_2[j]]
    # ②
    for j in range(l2):
        ci2[index_2_1[j]] = NP1[index_2_2[j]]
    l1 = len(index_2_2)
    l2 = len(index_1_2)
    for j in range(l1):
        ci1[index_2_2[j]] = ci1[index_2_2[j]] - 1
    for j in range(l2):
        ci2[index_1_2[j]] = ci2[index_1_2[j]] - 1
    NP1 = ci1
    NP2 = ci2

    s = [random.random() for _ in range(SH)]
    s = Tool.mylistRound(s)
    for i in range(0, SH):
        if (s[i] == 1):
            t = NM1[i]
            NM1[i] = NM2[i]
            NM2[i] = t
    return NP1, NM1, NF1, NP2, NM2, NF2

def crossover_FA(P1, M1, F1, P2, M2, F2, N, SH):
    # inital offerspring
    NP1 = P1
    NM1 = M1
    NP2 = P2
    NM2 = M2
    NF1 = F1
    NF2 = F2

    s = [random.random() for _ in range(N)]
    s = Tool.mylistRound(s)
    for i in range(0, N):
        if (s[i] == 1):
            t = NF1[i]
            NF1[i] = NF2[i]
            NF2[i] = t
    return NP1, NM1, NF1, NP2, NM2, NF2

def mutation_OSMS(p_chrom, m_chrom, SH, N, H, NM, M):
    p1 = math.floor(random.random() * SH)
    p2 = math.floor(random.random() * SH)
    while p1 == p2:
        p2 = math.floor(random.random() * SH)
    t = p_chrom[p1]
    p_chrom[p1] = p_chrom[p2]
    p_chrom[p2] = t
    s1 = p_chrom
    s2 = np.zeros(SH, dtype=int)
    p = np.zeros(N, dtype=int)
    fitness = np.zeros(2)
    for i in range(SH):
        p[s1[i]] = p[s1[i]] + 1
        s2[i] = p[s1[i]]
    s3 = m_chrom
    # 随机选两个位置
    p1 = math.floor(random.random() * SH)
    p2 = math.floor(random.random() * SH)
    while p1 == p2:
        p2 = math.floor(random.random() * SH)
    n = NM[s1[p1]][s2[p1] - 1]
    m = math.floor(random.random() * n)
    x = M[s1[p1]][s2[p1] - 1][m] - 1
    if n > 1:
        while s3[p1] == x:
            m = math.floor(random.random() * n)
            x = M[s1[p1]][s2[p1] - 1][m] - 1
    k1 = 0
    for t2 in range(s1[p1]):  # sum from 0 to s1[p1]
        k1 = k1 + H[t2]
    t1 = int(k1 + s2[p1] - 1)
    m_chrom[t1] = x

    n = NM[s1[p2]][s2[p2] - 1]
    m = math.floor(random.random() * n)
    x = M[s1[p2]][s2[p2] - 1][m] - 1
    if n > 1:
        while s3[p2] == x:
            m = math.floor(random.random() * n)
            x = M[s1[p2]][s2[p2] - 1][m] - 1
    k1 = 0
    for t2 in range(s1[p2]):  # sum from 0 to s1[p2]
        k1 = k1 + H[t2]
    t1 = int(k1 + s2[p2] - 1)
    m_chrom[t1] = x
    return p_chrom, m_chrom

def mutation_FA(p_chrom, m_chrom, f_chrom, SH, N, H, NM, M, F, TM):
    j1 = math.floor(random.random() * N)
    newj1factory = math.floor(random.random() * F)
    while newj1factory == f_chrom[j1]:
        newj1factory = math.floor(random.random() * F)
    f_chrom[j1] = newj1factory
    s1 = p_chrom
    s2 = np.zeros(SH, dtype=int)
    p = np.zeros(N, dtype=int)
    for i in range(SH):
        p[s1[i]] = p[s1[i]] + 1
        s2[i] = p[s1[i]]
    k1 = 0
    for t2 in range(j1):
        k1 = k1 + H[t2]

    for t2 in range(TM):
        n = NM[j1][t2]
        m = math.floor(random.random() * n)
        x = M[j1][t2][m] - 1
        m_chrom[k1 + t2] = x

    return p_chrom, m_chrom, f_chrom

def evolution_FA(p_chrom, m_chrom, f_chrom, index, Pc, Pm, ps, SH, N, H, NM, M, F, TM):
    R = math.floor(random.random() * ps)
    P1 = copy.copy(p_chrom[index, :])
    P2 = copy.copy(p_chrom[R, :])
    M1 = copy.copy(m_chrom[index, :])
    M2 = copy.copy(m_chrom[R, :])
    F1 = copy.copy(f_chrom[index, :])
    F2 = copy.copy(f_chrom[R, :])

    while R == index:
        R = math.floor(random.random() * ps)
    if random.random() < Pc:
        P1, M1, F1, P2, M2, F2 = crossover_FA(p_chrom[index, :], m_chrom[index, :], f_chrom[index, :], p_chrom[R, :],
                                           m_chrom[R, :], f_chrom[R, :], N, SH)
    if random.random() < Pm:
        P1, M1, F1 = mutation_FA(P1, M1, F1, SH, N, H, NM, M, F, TM)
        P2, M2, F2 = mutation_FA(P2, M2, F2, SH, N, H, NM, M, F, TM)

    return P1, M1, F1, P2, M2, F2

def evolution_OSMS(p_chrom, m_chrom, f_chrom, index, Pc, Pm, ps, SH, N, H, NM, M, F, TM):
    R = math.floor(random.random() * ps)
    P1 = copy.copy(p_chrom[index, :])
    P2 = copy.copy(p_chrom[R, :])
    M1 = copy.copy(m_chrom[index, :])
    M2 = copy.copy(m_chrom[R, :])
    F1 = copy.copy(f_chrom[index, :])
    F2 = copy.copy(f_chrom[R, :])
    while R == index:
        R = math.floor(random.random() * ps)
    if random.random() < Pc:
        P1, M1, F1, P2, M2, F2 = crossover_OSMS(p_chrom[index, :], m_chrom[index, :], f_chrom[index, :], p_chrom[R, :],
                                           m_chrom[R, :], f_chrom[R, :], N, SH)
    if random.random() < Pm:
        P1, M1 = mutation_OSMS(P1, M1, SH, N, H, NM, M)
        P2, M2 = mutation_OSMS(P2, M2, SH, N, H, NM, M)
    return P1, M1, F1, P2, M2, F2

def crossover(P1, M1, F1, P2, M2, F2, N, SH):
    NP1 = P1
    NM1 = M1
    NP2 = P2
    NM2 = M2
    NF1 = F1
    NF2 = F2
    ci1 = np.zeros(SH, dtype=int)
    ci2 = np.zeros(SH, dtype=int)
    temp = [random.random() for _ in range(N)]
    temp = Tool.mylistRound(temp)
    J1 = find_all_index(temp, 1)

    for j in range(SH):
        if Ismemeber(P1[j], J1) == 1:
            ci1[j] = P1[j] + 1
        if Ismemeber(P2[j], J1) == 0:
            ci2[j] = P2[j] + 1
    index_1_1 = find_all_index(ci1, 0)  # find the empty positions in ci1
    index_1_2 = find_all_index_not(ci2, 0)  # find the positions in ci2 which is not zero

    index_2_1 = find_all_index(ci2, 0)
    index_2_2 = find_all_index_not(ci1, 0)
    l1 = len(index_1_1)
    l2 = len(index_2_1)
    # ①
    for j in range(l1):
        ci1[index_1_1[j]] = NP2[index_1_2[j]]
    # ②
    for j in range(l2):
        ci2[index_2_1[j]] = NP1[index_2_2[j]]
    l1 = len(index_2_2)
    l2 = len(index_1_2)
    for j in range(l1):
        ci1[index_2_2[j]] = ci1[index_2_2[j]] - 1
    for j in range(l2):
        ci2[index_1_2[j]] = ci2[index_1_2[j]] - 1
    NP1 = ci1
    NP2 = ci2

    s = [random.random() for _ in range(SH)]
    s = Tool.mylistRound(s)
    for i in range(0, SH):
        if (s[i] == 1):
            t = NM1[i]
            NM1[i] = NM2[i]
            NM2[i] = t
    s = [random.random() for _ in range(N)]
    s = Tool.mylistRound(s)
    for i in range(0, N):
        if (s[i] == 1):
            t = NF1[i]
            NF1[i] = NF2[i]
            NF2[i] = t
    return NP1, NM1, NF1, NP2, NM2, NF2

def mutation(p_chrom, m_chrom, SH, N, H, NM, M):
    p1 = math.floor(random.random() * SH)
    p2 = math.floor(random.random() * SH)
    while p1 == p2:
        p2 = math.floor(random.random() * SH)
    t = p_chrom[p1]
    p_chrom[p1] = p_chrom[p2]
    p_chrom[p2] = t

    # change a machine for machine selection as mutation operator

    s1 = p_chrom
    s2 = np.zeros(SH, dtype=int)
    p = np.zeros(N, dtype=int)
    fitness = np.zeros(2)
    for i in range(SH):
        p[s1[i]] = p[s1[i]] + 1
        s2[i] = p[s1[i]]
    s3 = m_chrom
    p1 = math.floor(random.random() * SH)
    p2 = math.floor(random.random() * SH)
    while p1 == p2:
        p2 = math.floor(random.random() * SH)
    n = NM[s1[p1]][s2[p1] - 1]
    m = math.floor(random.random() * n)
    x = M[s1[p1]][s2[p1] - 1][m] - 1
    if n > 1:
        while s3[p1] == x:
            m = math.floor(random.random() * n)
            x = M[s1[p1]][s2[p1] - 1][m] - 1
    k1 = 0
    for t2 in range(s1[p1]):  # sum from 0 to s1[p1]
        k1 = k1 + H[t2]
    t1 = int(k1 + s2[p1] - 1)
    m_chrom[t1] = x

    n = NM[s1[p2]][s2[p2] - 1]
    m = math.floor(random.random() * n)
    x = M[s1[p2]][s2[p2] - 1][m] - 1
    if n > 1:
        while s3[p2] == x:
            m = math.floor(random.random() * n)
            x = M[s1[p2]][s2[p2] - 1][m] - 1
    k1 = 0
    for t2 in range(s1[p2]):  # sum from 0 to s1[p2]
        k1 = k1 + H[t2]
    t1 = int(k1 + s2[p2] - 1)
    m_chrom[t1] = x
    return p_chrom, m_chrom

def evolution(p_chrom, m_chrom, f_chrom, index, Pc, Pm, ps, SH, N, H, NM, M, F, TM):
    R = math.floor(random.random() * ps)
    P1 = copy.copy(p_chrom[index, :])
    P2 = copy.copy(p_chrom[R, :])
    M1 = copy.copy(m_chrom[index, :])
    M2 = copy.copy(m_chrom[R, :])
    F1 = copy.copy(f_chrom[index, :])
    F2 = copy.copy(f_chrom[R, :])
    while R == index:
        R = math.floor(random.random() * ps)
    if random.random() < Pc:
        P1, M1, F1, P2, M2, F2 = crossover(p_chrom[index, :], m_chrom[index, :], f_chrom[index, :], p_chrom[R, :],
                                           m_chrom[R, :], f_chrom[R, :], N, SH)
    if random.random() < Pm:
        P1, M1 = mutation(P1, M1, SH, N, H, NM, M)
        P2, M2 = mutation(P2, M2, SH, N, H, NM, M)
    return P1, M1, F1, P2, M2, F2

