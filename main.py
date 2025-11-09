# coding:utf-8
import copy
import math
import random

from DataRead import DataReadDHFJSP
import numpy as np
import os
from inital import initial
from fitFJSP import CalfitDHFJFP
from Tselection import *
from EA import *
from Tool import *
from FastNDSort import FastNDS
from EnergySave import EnergysavingDHFJSP
from LocalSearch import *
from test import *
import warnings

warnings.filterwarnings("error")

#
Combination = [[10, 2], [20, 2], [30, 2], [40, 2],
               [20, 3], [30, 3], [40, 3], [50, 3],
               [40, 4], [50, 4], [100, 4],
               [50, 5], [100, 5], [150, 5],
               [100, 6], [150, 6], [200, 6],
               [100, 7], [150, 7], [200, 7]]

datapath = 'DATASET/'
FileName = []
ResultPath = []
for i in range(20):
    J = Combination[i][0]
    F1 = Combination[i][1]
    O = 5
    temp = datapath + str(J) + 'J' + str(F1) + 'F' + '.txt'  # temp：数据集路径
    temp2 = str(J) + 'J' + str(F1) + 'F'  # temp2：与该数据集文件同名，用来创建一个文件夹，存储该数据集的结果
    FileName.append(temp)
    ResultPath.append(temp2)
TF = 20
FileName = np.array(FileName)
FileName.reshape(TF, 1)
ResultPath = np.array(ResultPath)
ResultPath.reshape(TF, 1)
# read the parameter of algorithm such as popsize, crossover rate, mutation rate
# 读参数
f = open("parameter.txt", "r", encoding  ='utf-8')
ps, Pc, Pm, epsilon, Theta= f.read().split(' ')
ps = int(ps)
Pc = float(Pc)
Pm = float(Pm)
epsilon = int(epsilon)
Theta = int(Theta)
f.close()
IndependentRun = 20

# execute algorithm for each instance
for file in range(0, 20):
    N, F, TM, H, SH, NM, M, time, ProF = DataReadDHFJSP(FileName[file])
    MaxNFEs = 200 * SH

    respath = 'result\\'
    respath2 = 'result\\'
    sprit = '\\'
    respath = respath + ResultPath[file]
    isExist = os.path.exists(respath)
    print(ResultPath[file], 'is being Optimizing\n')
    # start independent run for GMA
    for rround in range(10):
        p_chrom, m_chrom, f_chrom, group = initial(N, H, SH, NM, M, ps, F, time, TM)
        fitness = np.zeros(shape=(ps, 3))
        ftime = np.zeros(shape=(ps, F))
        finish = np.zeros(shape=(ps, N, 5))
        MT = np.zeros(shape=(ps, F, 5))
        NFEs = 0  # number of function evaluation
        # calucate fitness of each solution
        for i in range(ps):
            fitness[i, 0], fitness[i, 1], fitness[i, 2], ftime[i, :], finish[i, :], MT[i, :] = CalfitDHFJFP(
                p_chrom[i, :], m_chrom[i, :],
                f_chrom[i, :], N,
                H, SH, F, TM, time)

        # 精英档案
        AP = []
        AM = []
        AF = []
        AFit = []
        AFTime = []
        AFinish = []
        AMT = []
        Qual = []

        # 精英档案记录
        AP_record = []
        AM_record = []
        AF_record = []
        AFit_record = []
        AFTime_record = []
        AFinish_record = []
        AMT_record = []

        diftag = 0
        OSMSorFAflag = 0
        OSMSorFA = 0
        times = 1
        finaltag = 0
        transformationtime = 0
        while NFEs < MaxNFEs:
            print(FileName[file] + ' round ', rround + 1, 'iter ', times)
            times = times + 1

            ChildP = np.zeros(shape=(2 * ps, SH), dtype=int)
            ChildM = np.zeros(shape=(2 * ps, SH), dtype=int)
            ChildF = np.zeros(shape=(2 * ps, N), dtype=int)
            ChildFit = np.zeros(shape=(2 * ps, 3))
            ChildFtime = np.zeros(shape=(2 * ps, F))
            ChildFinish = np.zeros(shape=(2 * ps, N, 5))
            ChildMT = np.zeros(shape=(2 * ps, F, 5))
            # mating selection
            P_pool, M_pool, F_pool = tournamentSelection(p_chrom, m_chrom, f_chrom, fitness, ps, SH, N)
            # offspring generation
            for j in range(ps):
                Fit1 = np.zeros(3)
                Fit2 = np.zeros(3)
                if OSMSorFAflag < Theta:
                    P1, M1, F1, P2, M2, F2 = evolution_OSMS(P_pool, M_pool, F_pool, j, Pc, Pm, ps, SH, N, H, NM, M,
                                                        F, TM)
                else:
                    P1, M1, F1, P2, M2, F2 = evolution(P_pool, M_pool, F_pool, j, Pc, Pm, ps, SH, N, H, NM, M, F, TM)
                    OSMSorFA = 1
                Fit1[0], Fit1[1], Fit1[2], Ftime1, finish1, mt1 = CalfitDHFJFP(P1, M1, F1, N, H, SH, F, TM, time)
                Fit2[0], Fit2[1], Fit2[2], Ftime2, finish2, mt2 = CalfitDHFJFP(P2, M2, F2, N, H, SH, F, TM, time)
                NFEs = NFEs + 2
                t1 = j * 2
                t2 = j * 2 + 1
                ChildP[t1, :] = copy.copy(P1)
                ChildM[t1, :] = copy.copy(M1)
                ChildF[t1, :] = copy.copy(F1)
                ChildFit[t1, :] = Fit1
                ChildFtime[t1, :] = Ftime1
                ChildFinish[t1, :] = finish1
                ChildMT[t1, :] = mt1
                ChildP[t2, :] = copy.copy(P2)
                ChildM[t2, :] = copy.copy(M2)
                ChildF[t2, :] = copy.copy(F2)
                ChildFit[t2, :] = Fit2
                ChildFtime[t2, :] = Ftime2
                ChildFinish[t2, :] = finish2
                ChildMT[t2, :] = mt2

            if OSMSorFA:
                transformationtime = 1
                OSMSorFA = 0
                OSMSorFAflag = 0
                if finaltag == 0:
                    totalAFit = AFit
                    totalAP = AP
                    totalAM = AM
                    totalAF = AF
                    finaltag += 1
                else:
                    totalAFit = np.vstack((totalAFit, AFit))
                    totalAP = np.vstack((totalAP, AP))
                    totalAM = np.vstack((totalAM, AM))
                    totalAF = np.vstack((totalAF, AF))

                AP = []; AM = []; AF = []; AFit = []; AFTime = []; AFinish = []; AMT = []; Qual = []
                AP_record = []; AM_record = []; AF_record = []; AFit_record = []; AFTime_record = []
                AFinish_record = []; AMT_record = []
                diftag = 0

                QP, QM, QF, QFit, QFtime, QFinish, QMT = DeleteReapt(ChildP, ChildM, ChildF, ChildFit, ChildFtime, ChildFinish, ChildMT, ps)

                RQFit = QFit[:, 0:2]
                TopRank = FastNDS(RQFit, ps)

                p_chrom = QP[TopRank, :]
                m_chrom = QM[TopRank, :]
                f_chrom = QF[TopRank, :]
                fitness = QFit[TopRank, :]
                ftime = QFtime[TopRank, :]
                finish = QFinish[TopRank, :]
                MT = QMT[TopRank, :]
            else:
                QP = np.vstack((p_chrom, ChildP))
                QM = np.vstack((m_chrom, ChildM))
                QF = np.vstack((f_chrom, ChildF))
                QFit = np.vstack((fitness, ChildFit))
                QFtime = np.vstack((ftime, ChildFtime))
                QFinish = np.vstack((finish, ChildFinish))
                QMT = np.vstack((MT, ChildMT))

                QP, QM, QF, QFit, QFtime, QFinish, QMT = DeleteReapt(QP, QM, QF, QFit, QFtime, QFinish, QMT, ps)

                RQFit = QFit[:, 0:2]
                TopRank = FastNDS(RQFit, ps)

                p_chrom = QP[TopRank, :]
                m_chrom = QM[TopRank, :]
                f_chrom = QF[TopRank, :]
                fitness = QFit[TopRank, :]
                ftime = QFtime[TopRank, :]
                finish = QFinish[TopRank, :]
                MT = QMT[TopRank, :]

            # Elite strategy
            offspring_PF = pareto(fitness)

            if len(AFit) == 0:
                AP = p_chrom[offspring_PF, :]
                AM = m_chrom[offspring_PF, :]
                AF = f_chrom[offspring_PF, :]
                AFit = fitness[offspring_PF, :]
                AFTime = ftime[offspring_PF, :]
                AFinish = finish[offspring_PF, :]
                AMT = MT[offspring_PF, :]
            else:
                AP = np.vstack((AP, p_chrom[offspring_PF, :]))
                AM = np.vstack((AM, m_chrom[offspring_PF, :]))
                AF = np.vstack((AF, f_chrom[offspring_PF, :]))
                AFit = np.vstack((AFit, fitness[offspring_PF, :]))
                AFTime = np.vstack((AFTime, ftime[offspring_PF, :]))
                AFinish = np.vstack((AFinish, finish[offspring_PF, :]))
                AMT = np.vstack((AMT, MT[offspring_PF, :]))
                for j in range(len(offspring_PF)):
                    Qual = np.append(Qual, 1)


            if diftag == 0:
                L = len(AFit)
                Qual = np.ones(L)
            PF = pareto(AFit)

            AP = AP[PF, :]
            AM = AM[PF, :]
            AF = AF[PF, :]
            AFit = AFit[PF, :]
            AFinish = AFinish[PF, :]
            AMT = AMT[PF, :]
            Qual = Qual[PF]
            AP, AM, AF, AFit, AFTime, AFinish, AMT, Qual = (
                DeleteReaptE(AP, AM, AF, AFit, AFTime, AFinish, AMT, Qual))

            # Local search in Archive
            L = len(AFit)
            for l in range(L):
                current_AP = AP[l, :]
                current_AM = AM[l, :]
                current_AF = AF[l, :]
                current_AFit = AFit[l, :]
                current_AFTime = AFTime[l, :]
                current_finish = AFinish[l, :]
                current_AMT = AMT[l, :]
                if Qual[l] == 0:
                    continue
                localsearchtimes = 0

                max_localsearchtimes = epsilon * min(NFEs/(ps*20), 1)
                endflag = 0
                while localsearchtimes <= max_localsearchtimes:
                    localsearchtimes = localsearchtimes + 1
                    action = random.random() * 6
                    k = int(action)
                    if k == 0:
                        P1, M1, F1 = InsertRandOF(current_AP, current_AM, current_AF, current_AFit, N, H,
                                                  SH,
                                                  time, TM, NM, M,
                                                  F)
                    elif k == 1:
                        P1, M1, F1 = SwapICF(current_AP, current_AM, current_AF, current_AFTime,
                                             current_AFit,
                                             N, H, SH, time, TM,
                                             NM, M, F)
                    elif k == 2:
                        P1, M1, F1 = InsertRankOM(current_AP, current_AM, current_AF, current_AFit, N, H,
                                                  SH,
                                                  time, TM, NM, M, F, current_finish, current_AMT)
                    elif k == 3:
                        P1, M1, F1 = SwapOCF(current_AP, current_AM, current_AF, current_AFTime,
                                             current_AFit,
                                             N, H, SH, time, TM,
                                             NM, M, F)
                    elif k == 4:
                        P1, M1, F1 = SwapRankMS(current_AP, current_AM, current_AF, current_AFit,
                                                N, H, SH, time, TM, NM, M, F)
                    elif k == 5:
                        P1, M1, F1 = SwapRankOF(current_AP, current_AM, current_AF, current_AFTime, current_AFit,
                                        N, H, SH, time, TM, NM, M, F, current_finish, current_AMT)
                    Fit1[0], Fit1[1], Fit1[2], Ftime3, finish3, mt3 = CalfitDHFJFP(P1, M1, F1, N, H, SH, F,
                                                                                   TM, time)
                    NFEs = NFEs + 1
                    dom = NDS(Fit1, AFit[l, :])

                    if dom == 1:
                        AP[l, :] = copy.copy(P1)
                        AM[l, :] = copy.copy(M1)
                        AF[l, :] = copy.copy(F1)
                        AFit[l, :] = copy.copy(Fit1)
                        AFTime[l, :] = copy.copy(Ftime3)
                        AFinish[l, :] = copy.copy(finish3)
                        AMT[l, :] = copy.copy(mt3)

                        AP = np.vstack((AP, P1))
                        AM = np.vstack((AM, M1))
                        AF = np.vstack((AF, F1))
                        AFit = np.vstack((AFit, Fit1))
                        AFTime = np.vstack((AFTime, Ftime3))
                        finish3 = np.expand_dims(finish3, axis=0)
                        AFinish = np.vstack((AFinish, finish3))
                        mt3 = np.expand_dims(mt3, axis=0)
                        AMT = np.vstack((AMT, mt3))
                        Qual = np.append(Qual, 1)

                        front_number, front_no, max_fno = ndsort(fitness)
                        if random.random() < 0.5:
                            for a in range(ps):
                                if front_no[a] == max_fno:
                                    p_chrom[a] = P1
                                    m_chrom[a] = M1
                                    f_chrom[a] = F1
                        else:
                            tmp = int(random.random() * ps)
                            p_chrom[tmp] = P1
                            m_chrom[tmp] = M1
                            f_chrom[tmp] = F1

                        current_AP = P1
                        current_AM = M1
                        current_AF = F1
                        current_AFit = Fit1
                        current_AFTime = Ftime3

                    elif dom == 0 and AFit[l][0] != Fit1[0] and AFit[l][1] != Fit1[1]:
                        AP = np.vstack((AP, P1))
                        AM = np.vstack((AM, M1))
                        AF = np.vstack((AF, F1))
                        AFit = np.vstack((AFit, Fit1))
                        AFTime = np.vstack((AFTime, Ftime3))
                        finish3 = np.expand_dims(finish3, axis=0)
                        AFinish = np.vstack((AFinish, finish3))
                        mt3 = np.expand_dims(mt3, axis=0)
                        AMT = np.vstack((AMT, mt3))
                        Qual = np.append(Qual, 1)

                        front_number, front_no, max_fno = ndsort(fitness)

                        if random.random() < 0.5:
                            for a in range(ps):
                                if front_no[a] == max_fno:
                                    p_chrom[a] = P1
                                    m_chrom[a] = M1
                                    f_chrom[a] = F1
                        else:
                            tmp = int(random.random() * ps)
                            p_chrom[tmp] = P1
                            m_chrom[tmp] = M1
                            f_chrom[tmp] = F1

                        if random.random() < 0.5:
                            current_AP = P1
                            current_AM = M1
                            current_AF = F1
                            current_AFit = Fit1
                            current_AFTime = Ftime3
                    else:
                        endflag = endflag + 1
                        if endflag == 4:
                            Qual[l] = Qual[l] - 1
                            break


            # Energy save
            L = len(AFit)
            for j in range(L):
                P1, M1, F1 = EnergysavingDHFJSP(AP[j, :], AM[j, :], AF[j, :], AFit[j, :], N, H, TM,
                                                time, SH, F)
                Fit1[0], Fit1[1], Fit1[2], Ftime4, finish4, mt4 = CalfitDHFJFP(P1, M1, F1, N, H, SH, F, TM, time)
                NFEs = NFEs + 1
                if NDS(Fit1, AFit[j, :]) == 1:
                    AP[j, :] = copy.copy(P1)
                    AM[j, :] = copy.copy(M1)
                    AF[j, :] = copy.copy(F1)
                    AFit[j, :] = copy.copy(Fit1)
                    AFTime[j, :] = copy.copy(Ftime4)
                    AFinish[j, :] = copy.copy(finish4)
                    AMT[j, :] = copy.copy(mt4)
                    AP = np.vstack((AP, P1))
                    AM = np.vstack((AM, M1))
                    AF = np.vstack((AF, F1))
                    AFit = np.vstack((AFit, Fit1))
                    AFTime = np.vstack((AFTime, Ftime4))
                    finish4 = np.expand_dims(finish4, axis=0)
                    AFinish = np.vstack((AFinish, finish4))
                    mt4 = np.expand_dims(mt4, axis=0)
                    AMT = np.vstack((AMT, mt4))
                    Qual = np.append(Qual, 1)
                    front_number, front_no, max_fno = ndsort(fitness)
                    if random.random() < 0.5:
                        for a in range(ps):
                            if front_no[a] == max_fno:
                                p_chrom[a] = P1
                                m_chrom[a] = M1
                                f_chrom[a] = F1
                    else:
                        tmp = int(random.random() * ps)
                        p_chrom[tmp] = P1
                        m_chrom[tmp] = M1
                        f_chrom[tmp] = F1
                elif NDS(Fit1, AFit[j, :]) == 0:
                    AP = np.vstack((AP, P1))
                    AM = np.vstack((AM, M1))
                    AF = np.vstack((AF, F1))
                    AFit = np.vstack((AFit, Fit1))
                    AFTime = np.vstack((AFTime, Ftime4))
                    finish4 = np.expand_dims(finish4, axis=0)
                    AFinish = np.vstack((AFinish, finish4))
                    mt4 = np.expand_dims(mt4, axis=0)
                    AMT = np.vstack((AMT, mt4))
                    Qual = np.append(Qual, 1)
                    front_number, front_no, max_fno = ndsort(fitness)
                    if random.random() < 0.5:
                        for a in range(ps):
                            if front_no[a] == max_fno:
                                p_chrom[a] = P1
                                m_chrom[a] = M1
                                f_chrom[a] = F1
                    else:
                        tmp = int(random.random() * ps)
                        p_chrom[tmp] = P1
                        m_chrom[tmp] = M1
                        f_chrom[tmp] = F1

            AP, AM, AF, AFit, AFTime, AFinish, AMT, Qual = (
                DeleteReaptE(AP, AM, AF, AFit, AFTime, AFinish, AMT, Qual))

            if diftag == 0:
                diftag += 1
                AP_record = AP
                AM_record = AM
                AF_record = AF
                AFit_record = AFit
                AFinish_record = AFinish
                AMT_record = AMT
            else:
                combined_array = np.concatenate([AFit[:], AFit_record])
                max_value = np.max(combined_array, axis=0)
                min_value = np.min(combined_array, axis=0)

                if np.any(max_value[:2] == min_value[:2]):
                    OSMSorFAflag = OSMSorFAflag + 1
                else:
                    # 标准化
                    array_AFit = copy.copy(AFit[:, :2])
                    array_AFit_record = copy.copy(AFit_record[:, :2])
                    array_AFit = array_AFit.astype(float)
                    array_AFit_record = array_AFit_record.astype(float)
                    array_AFit[:, 0] = (array_AFit[:, 0] - min_value[0]) / (max_value[0] - min_value[0])
                    array_AFit[:, 1] = (array_AFit[:, 1] - min_value[1]) / (max_value[1] - min_value[1])
                    array_AFit_record[:, 0] = (array_AFit_record[:, 0] - min_value[0]) / (max_value[0] - min_value[0])
                    array_AFit_record[:, 1] = (array_AFit_record[:, 1] - min_value[1]) / (max_value[1] - min_value[1])

                    centroid1 = np.mean(array_AFit, axis=0)
                    centroid2 = np.mean(array_AFit_record, axis=0)
                    distance = math.sqrt(
                        math.pow(centroid1[0] - centroid2[0], 2) + math.pow(centroid1[1] - centroid2[1], 2)
                    )

                    if distance < math.sqrt(math.pow(0.05, 2) + math.pow(0.05, 2)):
                        OSMSorFAflag = OSMSorFAflag + 1
                    else:
                        OSMSorFAflag = 0
                        AP_record = AP
                        AM_record = AM
                        AF_record = AF
                        AFit_record = AFit
                        AFinish_record = AFinish
                        AMT_record = AMT

        # write elite solutions in txt
        totalAFit = AFit
        totalAP = AP
        totalAM = AM
        totalAF = AF
        for i in range(1, 10):
            totalAFit = np.vstack((totalAFit, AFit))
            totalAP = np.vstack((totalAP, AP))
            totalAM = np.vstack((totalAM, AM))
            totalAF = np.vstack((totalAF, AF))
        PF = pareto(totalAFit)
        totalAP = totalAP[PF, :]
        totalAM = totalAM[PF, :]
        totalAF = totalAF[PF, :]
        totalAFit = totalAFit[PF, :]
        PF = pareto(totalAFit)
        l = len(PF)
        obj = totalAFit[:, 0:2]
        newobj = []
        for i in range(l):
            newobj.append(obj[PF[i], :])
        newobj = np.unique(newobj, axis=0)  # delete the repeat row
        tmp = ResultPath[file] + 'res'

        resPATH = respath2 + tmp + str(rround + 1) + '.txt'
        f = open(resPATH, "w", encoding='utf-8')
        l = len(newobj)
        for i in range(l):
            item = '%5.2f %6.2f \n' % (newobj[i][0], newobj[i][1])  # fomat writing into txt file
            f.write(item)
        f.close()

    print('finish ' + FileName[file])

print('finish running')





