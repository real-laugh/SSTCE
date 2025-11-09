# coding:utf-8
#problem specific local search
import copy
import random
import math

import numpy as np
from CriticalPath import FindCriticalPathDHFJSP

def InsertRandOF(p_chrom, m_chrom, f_chrom, fitness, N, H, SH, time, TM, NM, M, F):
    s1 = p_chrom  # s1: 工序排序OS向量
    s2 = np.zeros(SH, dtype=int)  # s2: 1000×1的行向量
    p = np.zeros(N, dtype=int)  # p: 200×1行向量
    '''
    遍历OS向量 —— s1, s2记录 位置在OS中对应的工件编号的第几道工序
    p: 辅助记录, 每出现一次该工件编号, 则令p对应位置+1
    '''
    for i in range(SH):
        p[s1[i]] = p[s1[i]] + 1
        s2[i] = p[s1[i]]
    P0 = [];
    P = [];
    FJ = []
    # P和FJ均为元素数为F的列表, 每个元素也是一个列表
    for f in range(F):
        P.append([])
        FJ.append([])
    # 将所有工序分配到各个工厂中
    for i in range(SH):
        t1 = s1[i]
        t2 = s2[i]
        t3 = f_chrom[t1]
        P[t3].append(p_chrom[i])
    # 将所有工件分配到各个工厂中
    for i in range(N):
        t3 = f_chrom[i]
        FJ[t3].append(i)
    # cf: 关键工厂编号
    cf = int(fitness[2])
    # 在关键工厂中随机选择一个工件
    l = len(FJ[cf])
    index_j = int(np.floor(random.random() * l))
    # 随机选择一个工厂
    index_f = int(np.floor(random.random() * F))
    while index_f == cf:
        index_f = np.floor(random.random() * F)

    newf = copy.copy(f_chrom)
    newm = copy.copy(m_chrom)

    # print(f'FJ[cf] = {FJ[cf]}, type(FJ[cf]) = {type(FJ[cf])}')
    # 修改FA, 并返回
    newf[FJ[cf][index_j]] = index_f

    # 重新分配工厂后, 需要重新分配各工序选择的机器
    k1 = 0
    for i in range(FJ[cf][index_j]):
        k1 = k1 + H[i]
    for j in range(int(H[FJ[cf][index_j]])):
        # 随机选择一个机器  k2为机器的索引
        k2 = int(math.floor( random.random() * NM[ newf[FJ[cf][index_j]] * N + FJ[cf][index_j]][j] ))
        newm[k1 + j] = M[FJ[cf][index_j]][j][k2] - 1

    return p_chrom, newm, newf

def InsertRankOF(p_chrom, m_chrom, f_chrom, ftime, fitness, N, H, SH, time, TM, NM, M, F):
    s1 = p_chrom
    s2 = np.zeros(SH, dtype=int)
    p = np.zeros(N, dtype=int)
    for i in range(SH):
        p[s1[i]] = p[s1[i]] + 1
        s2[i] = p[s1[i]]
    P0 = []
    P = []
    FJ = []
    for f in range(F):
        P.append([])
        FJ.append([])
    for i in range(SH):
        t1 = s1[i]
        t2 = s2[i]
        t3 = f_chrom[t1]
        P[t3].append(p_chrom[i])

    for i in range(N):
        t3 = f_chrom[i]
        FJ[t3].append(i)

    cf = int(fitness[2])
    l = len(FJ[cf])
    index_j = int(np.floor(random.random() * l))
    minftime = ftime[0]
    index_f = 0
    for i in range(1, F):
        if ftime[i] < minftime:
            minftime = ftime[i]
            index_f = i

    newf = copy.copy(f_chrom)
    newf[FJ[cf][index_j]] = index_f

    newm = copy.copy(m_chrom)
    k1 = 0
    for i in range(FJ[cf][index_j]):
        k1 = k1 + H[i]
    for j in range(int(H[FJ[cf][index_j]])):
        k2 = int(math.floor(random.random() * NM[newf[FJ[cf][index_j]] * N + FJ[cf][index_j]][j]))
        newm[k1 + j] = M[FJ[cf][index_j]][j][k2] - 1

    return p_chrom, newm, newf

def SwapICF(p_chrom, m_chrom, f_chrom, ftime, fitness, N, H, SH, time, TM, NM, M, F):
    s1 = p_chrom
    s2 = np.zeros(SH, dtype=int)
    p = np.zeros(N, dtype=int)
    for i in range(SH):
        p[s1[i]] = p[s1[i]] + 1
        s2[i] = p[s1[i]]
    P0 = []
    P = []
    FJ = []
    for f in range(F):
        P.append([])
        FJ.append([])
        P0.append([])
    for i in range(SH):
        t1 = s1[i]
        t2 = s2[i]
        t3 = f_chrom[t1]
        P[t3].append(p_chrom[i])
        P0[t3].append(t2)
    for i in range(N):
        t3 = f_chrom[i]
        FJ[t3].append(i)
    cf = int(fitness[2])
    CP, _, _ = FindCriticalPathDHFJSP(P[cf], m_chrom, FJ[cf], cf, N, H, time, TM)
    L = len(CP)
    index0 = CP[int(np.floor(random.random() * L))]
    index1 = CP[int(np.floor(random.random() * L))]

    while P[cf][index1] == P[cf][index0]:
        index1 = CP[int(np.floor(random.random() * L))]

    index2 = P[cf][index0]
    index3 = P[cf][index1]
    index5 = 0
    index6 = 0

    newp = p_chrom
    newm = m_chrom
    newf = f_chrom
    j = P0[cf][index0]; k = P0[cf][index1]
    m0 = 0; m1 = 0
    for i in range(index2):
        m0 = m0 + H[i]
    for i in range(index3):
        m1 = m1 + H[i]
    m0 = m0 + j - 1; m1 = m1 + k - 1
    for i in range(SH):
        if newp[i] == index2:
            j = j - 1
            if j == 0:
                index5 = i
        if newp[i] == index3:
            k = k - 1
            if k == 0:
                index6 = i
        if j <= 0 and k <= 0:
            break

    temp = copy.copy(newp[index5])
    newp[index5] = copy.copy(newp[index6])
    newp[index6] = temp

    return newp, newm, newf

def InsertRankOM(p_chrom, m_chrom, f_chrom, fitness, N, H, SH, time, TM, NM, M, F, finish, MT):
    s1 = p_chrom
    s2 = np.zeros(SH, dtype=int)
    p = np.zeros(N, dtype=int)
    for i in range(SH):
        p[s1[i]] = p[s1[i]] + 1
        s2[i] = p[s1[i]]
    P0 = []
    P = []
    FJ = []
    MJ = []
    for f in range(F):
        P.append([])
        FJ.append([])
        P0.append([])
        MJ.append([])
        for i in range(TM):
            MJ[f].append([])
    for i in range(SH):
        t1 = s1[i]
        t2 = s2[i]
        t3 = f_chrom[t1]
        P[t3].append(p_chrom[i])
        P0[t3].append(t2)

        t4 = 0
        for k in range(t1):
            t4 = t4 + H[k]
        MJ[t3][m_chrom[t4 + t2 - 1]].append(t1)
    for i in range(N):
        t3 = f_chrom[i]
        FJ[t3].append(i)

    randF = int(random.random() * F)
    all_zeros = True
    while all_zeros:
        if not np.any(MT[randF]):
            randF = int(random.random() * F)
        else:
            all_zeros = False

    LM = 0
    for i in range(TM):
        if MT[randF][i] == max(MT[randF]):
            LM = i

    job = MJ[randF][LM][-1]
    for i in range(TM):
        if time[randF][job][H[job] - 1][i] != 0 and MT[randF][i] + time[randF][job][H[job] - 1][i] <= MT[randF][LM]:
            if MT[randF][i] >= finish[job][H[job] - 2]:
                t4 = 0
                for k in range(job):
                    t4 = t4 + H[k]
                m_chrom[t4 + H[job] - 1] = i

    return p_chrom, m_chrom, f_chrom

def SwapRankOF(p_chrom, m_chrom, f_chrom, ftime, fitness, N, H, SH, time, TM, NM, M, F, finish, MT):
    cf = int(fitness[2])

    FCT, cm = max(MT[cf]), MT[cf].argmax()
    CMJ = []
    s1 = p_chrom
    s2 = np.zeros(SH, dtype=int)
    p = np.zeros(N, dtype=int)
    for i in range(SH):
        p[s1[i]] = p[s1[i]] + 1
        s2[i] = p[s1[i]]
    P0 = []
    P = []
    FJ = []
    for f in range(F):
        P.append([])
        FJ.append([])

    for i in range(SH):
        t1 = s1[i]
        t2 = s2[i]
        t3 = f_chrom[t1]
        P[t3].append(p_chrom[i])
        t4 = 0
        for k in range(t1):
            t4 = t4 + H[k]
        if t3 == cf and m_chrom[t4 + t2 - 1] == cm:
            CMJ.append(t1)

    for i in range(N):
        t3 = f_chrom[i]
        FJ[t3].append(i)

    unique_CMJ = []
    for item in CMJ:
        if item not in unique_CMJ:
            unique_CMJ.append(item)
    CMJ = unique_CMJ
    maxtime = 0
    for i in range(len(CMJ)):
        currentTime = 0
        for j in range(H[CMJ[i]]):
            currentTime = currentTime + time[cf][CMJ[i]][j][m_chrom[5*CMJ[i] + j]]
        if currentTime > maxtime:
            maxtime = currentTime
            jobNum = CMJ[i]

    newF = -1
    for f in range(F):
        end = 0
        if f == cf:
            continue
        for m in range(TM):
            MCT = MT[f][m]
            for h in range(H[jobNum]):
                if m not in M[jobNum][h]:
                    continue
                MCT = MCT + time[f][jobNum][h][m]
            if MCT > FCT:
                end = 1
                continue

        if end == 1:
            continue
        else:
            newF = f
            break
    if newF >= 0:
        f_chrom[jobNum] = newF
    else:
        p_chrom, m_chrom, f_chrom = InsertRankOF(p_chrom, m_chrom, f_chrom, ftime, fitness, N, H, SH, time, TM, NM, M, F)

    return p_chrom, m_chrom, f_chrom

def SwapOCF(p_chrom, m_chrom, f_chrom, ftime, fitness, N, H, SH, time, TM, NM, M, F):
    s1 = p_chrom
    s2 = np.zeros(SH, dtype=int)
    p = np.zeros(N, dtype=int)
    for i in range(SH):
        p[s1[i]] = p[s1[i]] + 1
        s2[i] = p[s1[i]]
    P0 = []
    P = []
    FJ = []
    for f in range(F):
        P.append([])
        FJ.append([])
        P0.append([])
    for i in range(SH):
        t1 = s1[i]
        t2 = s2[i]
        t3 = f_chrom[t1]
        P[t3].append(p_chrom[i])
        P0[t3].append(t2)
    # 将所有工件分配到各个工厂中
    for i in range(N):
        t3 = f_chrom[i]
        FJ[t3].append(i)

    cf = int(fitness[2])
    CP, _, _ = FindCriticalPathDHFJSP(P[cf], m_chrom, FJ[cf], cf, N, H, time, TM)
    L = len(CP)
    index0 = CP[int(np.floor(random.random() * L))]

    index1 = P[cf][index0]
    index3 = 0

    newp = p_chrom
    newm = m_chrom
    newf = f_chrom
    j = P0[cf][index0]
    m0 = 0; m1 = 0
    for i in range(index1):
        m0 = m0 + H[i]
    m0 = m0 + j - 1
    for i in range(SH):
        if newp[i] == index1:
            j = j - 1
            if j == 0:
                index3 = i
        if j <= 0:
            break

    index4 = int(random.random() * SH)
    while index4 == index3:
        index4 = int(random.random() * SH)

    temp = copy.copy(newp[index3])
    newp[index3] = copy.copy(newp[index4])
    newp[index4] = temp

    return newp, newm, newf

#Randomly select a operation and assiged to another machine
def SwapRankMS(p_chrom,m_chrom,f_chrom,fitness,N,H,SH,time,TM,NM,M,F):
    s1 = p_chrom
    s2 = np.zeros(SH, dtype=int)
    p = np.zeros(N, dtype=int)
    for i in range(SH):
        p[s1[i]] = p[s1[i]] + 1
        s2[i] = p[s1[i]]
    P0 = []
    P = []
    FJ = []
    for f in range(F):
        P.append([])
        FJ.append([])

    for i in range(SH):
        t1 = s1[i]
        t2 = s2[i]
        t3 = f_chrom[t1]
        P[t3].append(p_chrom[i])
    for i in range(N):
        t3 = f_chrom[i]
        FJ[t3].append(i)

    cf = int(fitness[2])
    CP, _, _ = FindCriticalPathDHFJSP(P[cf], m_chrom, FJ[cf], cf, N, H, time, TM)
    L = len(CP)
    IndexO = CP[int(np.floor(random.random() * L))]
    s3 = copy.copy(P[cf])
    L = len(s3)
    s4=np.zeros(L)
    p2=np.zeros(N)
    for i in range(L):
        p2[s3[i]] = p2[s3[i]] + 1
        s4[i] = p2[s3[i]]
    I2=s3[IndexO]
    J2=s4[IndexO]
    for i in range(SH):
        if s1[i]==I2 and s2[i]==J2:
            IndexO=i
            break
    I=s1[IndexO];J=s2[IndexO]
    newm=m_chrom
    t4 = 0;t1=I;t2=J;
    for k in range(t1):  # sum from 0 to t1-1
        t4 = t4 + H[k]
    tmp = m_chrom[t4 + t2 - 1]
    n=NM[I][J-1]
    cm=np.zeros(n,dtype=int)
    cmt=np.zeros(n)
    tot=0
    for kk in range(n):
        cm[kk]=M[I][J-1][kk]-1
        cmt[kk]=time[cf][I][J-1][cm[kk]]
        tot=cmt[kk]+tot
    for kk in range(n):
        cmt[kk]=cmt[kk]/tot
    pro_index=np.argsort(cmt)
    pro = copy.copy(cmt[pro_index])
    for f in range(1,n):
        pro[f]=pro[f]+pro[f-1]
    x=random.random()
    for f in range(n):
        if x<pro[f]:
            Index2=cm[pro_index[f]]
            break

    while tmp==Index2:
        x = random.random()
        for f in range(n):
            if x < pro[f]:
                Index2 = cm[pro_index[f]]
                break
    newm[t4 + t2 - 1]=int(Index2)
    newp=p_chrom
    return p_chrom, m_chrom, f_chrom
