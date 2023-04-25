import numpy as np
import ast
from gurobipy import *
import time
import math


class StochIPinst:
    def __init__(self, instance):
        pass

    # Generating Benders cuts
    def addBenders(self, init=False, method='cutpl', tol=1e-4, timeLimit=60 * 60, store=False, ifBnC=False):
        if init == False:
            wrtStr = 'method=' + str(method) + '\ttol=' + str(tol) + '\ttimeLimit=' + str(timeLimit) + '\n'
            if ifBnC == False:
                fileName = "Results/" + str(self.name) + "_Benders.txt"
            else:
                fileName = "Results/" + str(self.name) + "_BendersBnC.txt"
            f = open(fileName, "a")
            f.write(wrtStr)
            f.close

        t0 = time.time()
        TimeMasterLP = 0.0
        TimeCutLP = None
        TimeCutQP = None
        TimeBaseMIP = None
        TimeSub = None
        AvgBaseSize = None
        NscenSolved = None
        MaxTime = None
        MinTime = None
        MaxSize = None
        MinSize = None
        x_value = {}                                    # current Master problem, optimal x value
        theta_value = {}
        if init == True:
            self.PrimalMaster.setObjective(0.0)
            self.PrimalMaster.update()
            self.PrimalMaster.optimize()
            for i in range(self.Nfv):
                x_value[i] = self.x[i].x
            for s in range(self.Nscen):
                theta_value[s] = -float('inf')
            self.PrimalMaster.setObjective(quicksum(self.cVec[i] * self.x[i] for i in range(self.Nfv)) + quicksum(
                self.pVec[s] * self.theta[s] for s in range(self.Nscen)))
            self.PrimalMaster.update()
        else:
            self.PrimalMaster.update()
            tStart = time.time()
            self.PrimalMaster.optimize()
            TimeMasterLP = TimeMasterLP + (time.time() - tStart)                        # Master problem will be solved multiple times, we add all time together
            for i in range(self.Nfv):
                x_value[i] = self.x[i].x
            for s in range(self.Nscen):
                theta_value[s] = self.theta[s].x

        BendersUB = float('inf')
        LB = -float('inf')
        ContinueCondition = True
        iter = 0

        while ContinueCondition == True and time.time() - t0 < timeLimit:
            iter += 1
            NscenSolved = 0
            MaxTime = -float('inf')                                             # combine with the following code, we know that we want to record the maximum time to solve a scenario
            MinTime = float('inf')
            BendersCutsAdded = 0
            ContinueCondition = False   
            CurObj = sum(self.cVec[i] * x_value[i] for i in range(self.Nfv))    # current objective
            for s in range(self.Nscen):
                NscenSolved = NscenSolved + 1
                tScen = time.time()
                ObjV, const, subg = self.SolveBendersSub(scen_id=s, x_input=x_value)
                if theta_value[s] < ObjV - tol * (abs(theta_value[s]) + 1) or init == True:
                    # for each scenario, we have a cutList (initiated in snipMaster.readData)
                    self.thetaCutList[s].append( # we append the cut
                        self.PrimalMaster.addConstr(self.theta[s] >= const + quicksum(subg[i] * self.x[i] for i in range(self.Nfv)))  # we add a Benders cut into the Master problem
                        )
                    self.Ncuts += 1                                 # this records all cuts cross scenarios
                    BendersCutsAdded += 1                           # this only records the cuts in this scenario
                    self.cutlist[s].append(subg)
                    coef = subg.copy()
                    coef[self.Nfv] = 1
                    self.coeflist[s].append(coef)
                    if method == 'cutpl':
                        ContinueCondition = True                    # when all scenarios have been well-approximated, the procedures will not go into here
                CurObj = CurObj + self.pVec[s] * ObjV               # this is an upper bound (from an incumbent)
                tScen = time.time() - tScen                         # time to solve this scenario
                if tScen > MaxTime:
                    MaxTime = tScen
                if tScen < MinTime:
                    MinTime = tScen
            # each scenarios has been solved and added a cut to approximate it

            # Now, solve the master problem to see what is the LB now
            self.PrimalMaster.update()
            tStart = time.time()
            self.PrimalMaster.optimize()
            TimeMasterLP = TimeMasterLP + (time.time() - tStart)
            LB = self.PrimalMaster.objval
            for s in range(self.Nscen):
                theta_value[s] = self.theta[s].x                    # here we update the approximation of value function, in the next round, if theta < BbjV, then add cut

            if CurObj < BendersUB:                                  # update UB
                BendersUB = CurObj

            if method == 'level' and BendersUB > LB + tol * (abs(LB) + 1):
                if iter == 1 or sum(abs(x_value[i] - x_valueOld[i]) for i in range(self.Nfv)) > 1e-5:
                    ContinueCondition = True

            if method == 'cutpl' and ContinueCondition == True:
                for j in range(self.Nfv):
                    x_value[j] = self.x[j].x

            if method == 'level' and ContinueCondition == True:
                lt = LB + 0.3 * (BendersUB - LB)
                ltConstr = self.PrimalMaster.addConstr(
                    quicksum(self.pVec[s] * self.theta[s] for s in range(self.Nscen)) + quicksum(
                        self.cVec[j] * self.x[j] for j in range(self.Nfv)) <= lt)
                self.PrimalMaster.setObjective(
                    quicksum((self.x[j] - x_value[j]) * (self.x[j] - x_value[j]) for j in range(self.Nfv)))
                self.PrimalMaster.update()
                self.PrimalMaster.optimize()
                x_valueOld = x_value.copy()
                for i in range(self.Nfv):
                    x_value[i] = self.x[i].x
                self.PrimalMaster.remove(ltConstr)
                self.PrimalMaster.setObjective(
                    quicksum(self.pVec[s] * self.theta[s] for s in range(self.Nscen)) + quicksum(
                        self.cVec[j] * self.x[j] for j in range(self.Nfv)))
                self.PrimalMaster.update()

            print('Benders Iter ' + str(iter) + ', PrimalMaster LB: ' + str(LB) + ', UB: ' + str(
                BendersUB) + ', Cuts Added: ' + str(BendersCutsAdded))

            # truncate the Benders process in 'initiate' method
            if init == True and iter >= 1:                                  
                ContinueCondition = False

            if store == True:
                self.PrimalMaster.update()
                tStart = time.time()
                self.PrimalMaster.optimize()
                TimeMasterLP = TimeMasterLP + (time.time() - tStart)
                LB = self.PrimalMaster.objval
                wrtStr = str(iter) + '\t' + str(time.time() - t0) + '\t' + str(LB) + '\t' + str(
                    TimeMasterLP) + '\t' + str(TimeCutLP) + '\t' + str(TimeCutQP) + '\t' + str(TimeBaseMIP) + \
                         '\t' + str(TimeSub) + '\t' + str(AvgBaseSize) + '\t' + str(NscenSolved) + '\t' + str(
                    MaxTime) + '\t' + str(MinTime) + '\t' + str(MaxSize) + '\t' + str(MinSize) + '\t' + \
                         str(method) + '\t' + str(self.Ncuts) + '\t' + str(self.Nsubs) + '\n'
                f = open(fileName, "a")
                f.write(wrtStr)
                f.close
        return TimeMasterLP

    # Generating strengthened Benders cuts
    def PureStrBenders(self, BDmethod='level', tol=1e-4, timeLimit=60 * 60):
        wrtStr = 'BendersMethod=' + str(BDmethod) + '\ttol=' + str(tol) + '\ttimeLimit=' + str(timeLimit) + '\n'
        f = open("Results/" + str(self.name) + "_StrBenders.txt", "a")
        f.write(wrtStr)
        f.close
        t0 = time.time()
        TimeMasterLP = 0.0
        TimeCutLP = None
        TimeCutQP = None
        TimeBaseMIP = None
        TimeSub = 0.0
        AvgBaseSize = None
        NscenSolved = None
        MaxTime = None
        MinTime = None
        MaxSize = None
        MinSize = None
        TimeMasterLP = TimeMasterLP + self.addBenders(self, method=BDmethod, timeLimit=timeLimit - (time.time() - t0))
        x_value = {}
        theta_value = {}
        ContinueCondition = True
        iter = 0
        while ContinueCondition == True and time.time() - t0 < timeLimit:
            iter += 1
            NscenSolved = 0
            MaxTime = -float('inf')
            MinTime = float('inf')
            ContinueCondition = False
            self.PrimalMaster.update()
            tStart = time.time()
            self.PrimalMaster.optimize()
            TimeMasterLP = TimeMasterLP + (time.time() - tStart)
            LB = self.PrimalMaster.objval
            for i in range(self.Nfv):
                x_value[i] = self.x[i].x
            for s in range(self.Nscen):
                theta_value[s] = self.theta[s].x
            for s in range(self.Nscen):
                NscenSolved = NscenSolved + 1
                tScen = time.time()
                ObjV, const, subg = self.SolveBendersSub(scen_id=s, x_input=x_value)                            # subg is the negative dual multiplier (pi / lambda)
                nsubg = {i: -subg[i] for i in subg}                                                             # obtain the dual multipliers
                tSub = time.time()
                tlimitValue = max(timeLimit - (time.time() - t0), 0)
                ObjV, xHat, yObjV, SubOpt, BestBound = self.SolveScenSub(scen_id=s, objCoef=nsubg, regCoefy=1,  # we input the dual multiplier
                                                                         tlimit=tlimitValue)
                TimeSub = TimeSub + (time.time() - tSub)
                if theta_value[s] < sum(subg[i] * x_value[i] for i in range(self.Nfv)) + ObjV - tol * (
                        abs(theta_value[s]) + 1):
                    self.PrimalMaster.addConstr(
                        self.theta[s] >= quicksum(subg[i] * self.x[i] for i in range(self.Nfv)) + ObjV)
                    self.Ncuts += 1
                    ContinueCondition = True                                                                    # if there is a scenario added a cut, we will re-run the master problem and examine whether we need a new cut
                tScen = time.time() - tScen
                if tScen > MaxTime:
                    MaxTime = tScen
                if tScen < MinTime:
                    MinTime = tScen
            print('StrBenders Iter ' + str(iter) + ', PrimalMaster LB: ' + str(LB) + ', Total Time: ' + str(
                time.time() - t0) + ', Ncuts: ' + str(self.Ncuts))

            self.PrimalMaster.update()
            tStart = time.time()
            self.PrimalMaster.optimize()
            TimeMasterLP = TimeMasterLP + (time.time() - tStart)
            LB = self.PrimalMaster.objval
            wrtStr = str(iter) + '\t' + str(time.time() - t0) + '\t' + str(LB) + '\t' + str(TimeMasterLP) + '\t' + str(
                TimeCutLP) + '\t' + str(TimeCutQP) + '\t' + str(TimeBaseMIP) + \
                     '\t' + str(TimeSub) + '\t' + str(AvgBaseSize) + '\t' + str(NscenSolved) + '\t' + str(
                MaxTime) + '\t' + str(MinTime) + '\t' + str(MaxSize) + '\t' + str(MinSize) + '\t' + \
                     str(BDmethod) + '\t' + str(self.Ncuts) + '\t' + str(self.Nsubs) + '\n'
            f = open("Results/" + str(self.name) + "_StrBenders.txt", "a")
            f.write(wrtStr)
            f.close

    # Exact separation of Lagrangian cuts
    def IterativeLag(self, BDmethod='level', tol=1e-4, gapTol=5e-1, pi0Coef=1e-2, timeLimit=60 * 60):       # pi0Coef is the alpha in (19
        # Solved by level method (vanilla cutting plane is much slower for high dimensional problems)
        wrtStr = 'BendersMethod=' + str(BDmethod) + '\ttol=' + str(tol) + '\tgapTol=' + str(
            gapTol) + '\tpi0Coef=' + str(pi0Coef) + '\ttimeLimit=' + str(timeLimit) + '\n'
        f = open("Results/" + str(self.name) + "_IterLag_" + str(gapTol) + "_" + str(pi0Coef) + ".txt", "a")
        f.write(wrtStr)
        f.close
        t0 = time.time()
        TimeMasterLP = 0.0
        TimeCutLP = 0.0
        TimeCutQP = 0.0
        TimeBaseMIP = None
        TimeSub = 0.0
        AvgBaseSize = None
        NscenSolved = None
        TimeMasterLP = TimeMasterLP + self.addBenders(self, method=BDmethod)
        x_value = {}
        theta_value = {}
        iter = 0
        CutHistory = {}
        for s in range(self.Nscen):
            CutHistory[s] = []

        # Add perfect information soln
        piHat = {}
        subg = {}
        for i in range(self.Nfv):
            piHat[i] = self.cVec[i]
            subg[i] = -self.cVec[i]
        pi0Hat = 1.0
        for s in range(self.Nscen):
            tStart = time.time()
            tlimitValue = max(timeLimit - (time.time() - t0), 0)
            ObjV, xHat, yObjV, SubOpt, BestBound = self.SolveScenSub(scen_id=s, objCoef=piHat, regCoefy=pi0Hat,
                                                                     tlimit=tlimitValue)
            TimeSub = TimeSub + (time.time() - tStart)

            coef = [xHat[i] for i in range(self.Nfv)]
            coef.append(yObjV)
            CutHistory[s].append(coef)                                          # this is the cut for level method to approximate the dual problem
            for soln in SubOpt:
                coef = [soln[i] for i in range(self.Nfv + 1)]
                CutHistory[s].append(coef)

        FindCut = True
        while FindCut == True and time.time() - t0 < timeLimit:                 # the iteration of Backward and Forward
            MaxTime = -float('inf')
            MinTime = float('inf')
            MaxSize = None
            MinSize = None
            iter += 1
            NscenSolved = 0
            self.PrimalMaster.update()
            tStart = time.time()
            self.PrimalMaster.optimize()
            TimeMasterLP = TimeMasterLP + (time.time() - tStart)
            print('IterLag. Iter ' + str(iter) + ', PrimalMaster LB: ' + str(
                self.PrimalMaster.objval) + ', Total Time: ' + str(time.time() - t0) + ', Ncuts: ' + str(self.Ncuts))
            FindCut = False
            x_value = {}
            theta_value = {}
            for j in range(self.Nfv):
                x_value[j] = self.x[j].x
            for s in range(self.Nscen):
                theta_value[s] = self.theta[s].x

            for s in range(self.Nscen):                                                                             # the iteration of Forward for each scenario and Backward for each scenario
                NscenSolved = NscenSolved + 1
                if time.time() - t0 > timeLimit:
                    break
                tScen = time.time()
                LagIter = 0
                scenmax = Model('scenmax')                                                                          # the problem (17) in the paper
                piScenmax = {}
                absPiScenmax = {}
                pi0 = scenmax.addVar(lb=0.0)
                lpi = scenmax.addVar(lb=-GRB.INFINITY)                                                              # lpi is the linear approximation of Q_s(pi, pi0)
                for i in range(self.Nfv):
                    piScenmax[i] = scenmax.addVar(lb=-GRB.INFINITY)
                    absPiScenmax[i] = scenmax.addVar(lb=0.0)
                    scenmax.addConstr(piScenmax[i] <= absPiScenmax[i])
                    scenmax.addConstr(-piScenmax[i] <= absPiScenmax[i])
                scenmax.addConstr(quicksum(absPiScenmax[i] for i in range(self.Nfv)) + pi0Coef * pi0 <= 1)          # we only restrict the feasible region of (pi0, pi) by regularization condition, that is, |pi| + alpha * pi0 <= 1
                scenmax.setObjective(
                    lpi -quicksum(piScenmax[i] * x_value[i] for i in range(self.Nfv)) - pi0 * theta_value[s]
                                        )
                scenmax.modelSense = GRB.MAXIMIZE
                scenmax.setParam('OutputFlag', False)

                # Add stored solutions
                for coef in CutHistory[s]:
                    scenmax.addConstr(
                        lpi <= quicksum(coef[i] * piScenmax[i] for i in range(self.Nfv)) + coef[self.Nfv] * pi0
                                            )
                scenmax.update()
                tStart = time.time()
                scenmax.optimize()
                TimeCutLP = TimeCutLP + (time.time() - tStart)

                ContinueCondition = True
                piHat = {}
                for i in range(self.Nfv):
                    piHat[i] = piScenmax[i].x
                pi0Hat = pi0.x
                piBest = None
                pi0Best = None
                lBest = None
                LB = -float('inf')                                                                                      # Lower bound of level method Gap
                UB = float('inf')
                lpiold = float('inf')

                while ContinueCondition == True and time.time() < t0 + timeLimit:                                       # the level-set method iteration to find the best dual multipliers
                    LagIter += 1
                    if UB < tol * (abs(theta_value[s]) + 1):
                        print('scenario ' + str(s) + ': UB < tol, total time: ' + str(time.time() - t0))
                        break
                    tStart = time.time()
                    tlimitValue = max(timeLimit - (time.time() - t0), 0)
                    ObjV, xHat, yObjV, SubOpt, BestBound = self.SolveScenSub(scen_id=s, objCoef=piHat, regCoefy=pi0Hat,   # this is the problem Q(pi, pi0)
                                                                             tlimit=tlimitValue) 
                    TimeSub = TimeSub + (time.time() - tStart)

                    # every time, we collect all optimal and suboptimal solution from the solver to generate the cutting-plane approximation
                    coef = [xHat[i] for i in range(self.Nfv)]
                    coef.append(yObjV)
                    CutHistory[s].append(coef)
                    scenmax.addConstr(lpi <= quicksum(xHat[i] * piScenmax[i] for i in range(self.Nfv)) + yObjV * pi0)
                    for soln in SubOpt:
                        coef = [soln[i] for i in range(self.Nfv + 1)]
                        CutHistory[s].append(coef)
                        scenmax.addConstr(
                            lpi <= quicksum(soln[i] * piScenmax[i] for i in range(self.Nfv)) + soln[self.Nfv] * pi0)

                    gap = ObjV - pi0Hat * theta_value[s] - sum(piHat[i] * x_value[i] for i in range(self.Nfv))           # this is a LB (feasible point)
                    if gap > LB:                                                                                         # find a better LB
                        LB = gap
                        piBest = piHat.copy()                                                                            # then we update incumbent
                        pi0Best = pi0Hat
                        lBest = ObjV

                    scenmax.update()
                    tStart = time.time()
                    scenmax.optimize()
                    TimeCutLP = TimeCutLP + (time.time() - tStart)
                    UB = scenmax.objval                                                                                   # UB from the optimal solution of relaxed problem

                    if LagIter % 100 == 0:
                        print('IterLag. Cut iter: ' + str(LagIter) + ', scenario: ' + str(s) + ', UB: ' + str(
                            UB) + ', LB: ' + str(LB) + ', pi0: ' + str(pi0Hat))
                    if UB - LB < gapTol * UB or UB - LB < 1e-6 or LagIter > 1000:
                        if pi0Best > 1e-6:
                            print(
                                'IterLag. Cut iter: ' + str(LagIter) + ', scenario: ' + str(s) + ', violation: ' + str(
                                    LB / pi0Best) + ', total time: ' + str(time.time() - t0))
                        else:
                            print('IterLag. Cut iter: ' + str(LagIter) + ', scenario: ' + str(
                                s) + ', pi0Hat <= 1e-6, total time: ' + str(time.time() - t0))
                        ContinueCondition = False
                        if pi0Best > 1e-6 and LB / pi0Best >= tol * (abs(theta_value[s]) + 1):
                            # note that here is added the cut for Master problem
                            self.PrimalMaster.addConstr(pi0Best * self.theta[s] >= -quicksum(
                                piBest[i] * self.x[i] for i in range(self.Nfv)) + lBest)
                            self.Ncuts += 1
                            subg = {}
                            for i in range(self.Nfv):
                                subg[i] = -piBest[i] / pi0Best
                            FindCut = True                                                          # the parameter FindCut is to control whether we will have a new iteration from the very beginning
                    else:
                        QPsolved = True
                        lt = UB - 0.3 * (UB - LB)
                        ltConstr = scenmax.addConstr(
                            -quicksum(piScenmax[i] * x_value[i] for i in range(self.Nfv)) + lpi - pi0 * theta_value[
                                s] >= lt)
                        scenmax.setObjective(
                            quicksum((piScenmax[i] - piHat[i]) * (piScenmax[i] - piHat[i]) for i in range(self.Nfv)) + (
                                        pi0 - pi0Hat) * (pi0 - pi0Hat))
                        scenmax.modelSense = GRB.MINIMIZE
                        scenmax.params.Method = 2           # interior-point method for QP
                        scenmax.update()
                        tStart = time.time()
                        scenmax.optimize()
                        if scenmax.status != 2:
                            print('QP status: ' + str(scenmax.status) + ' with Method=' + str(
                                scenmax.params.Method) + '... Switching to 1')
                            scenmax.params.Method = 1                                               # model.params.Method = - 1, automatically find the suitable method; 
                            scenmax.update()                                                        # model.params.Method = 0, primal simplex; model.params.Method = 1, dual simplex; 
                            scenmax.optimize()                                                      # model.params.Method = 2, interior-point method for QP; model.params.Method = 3, parallel;...
                            if scenmax.status != 2:
                                print('QP status: ' + str(scenmax.status) + ' with Method=' + str(
                                    scenmax.params.Method) + '... Switching to 0')
                                scenmax.params.Method = 0
                                scenmax.update()
                                scenmax.optimize()
                                if scenmax.status != 2:
                                    print('QP status: ' + str(scenmax.status) + ' with Method=' + str(
                                        scenmax.params.Method) + '... Stop!')
                                    QPsolved = False
                        TimeCutQP = TimeCutQP + (time.time() - tStart)
                        piHatOld = piHat.copy()
                        pi0HatOld = pi0Hat
                        for i in range(self.Nfv):
                            piHat[i] = piScenmax[i].x                                               # we use bundle method to update the multipliers
                        pi0Hat = pi0.x
                        if QPsolved == False or (
                                max(abs(piHat[i] - piHatOld[i]) for i in range(self.Nfv)) < 1e-10 and abs(
                                pi0Hat - pi0HatOld) < 1e-10 and abs(lpi.x - lpiold) < 1e-10):
                            print('Same Solution/QP not solved!')
                            if pi0Best > 1e-6:
                                print('IterLag. Cut iter: ' + str(LagIter) + ', scenario: ' + str(
                                    s) + ', violation: ' + str(LB / pi0Best) + ', total time: ' + str(time.time() - t0))
                            ContinueCondition = False
                            if pi0Best > 1e-6 and LB >= tol * (abs(theta_value[s]) + 1):
                                # Add the cut for Master problem
                                self.PrimalMaster.addConstr(pi0Best * self.theta[s] >= -quicksum(
                                    piBest[i] * self.x[i] for i in range(self.Nfv)) + lBest)
                                self.Ncuts += 1
                                FindCut = True
                        lpiold = lpi.x
                        scenmax.remove(ltConstr)
                        scenmax.setObjective(
                                    lpi - quicksum(piScenmax[i] * x_value[i] for i in range(self.Nfv)) - pi0 * theta_value[s]
                                                )
                        scenmax.modelSense = GRB.MAXIMIZE
                        scenmax.params.Method = -1                                                   # change to automatically find a suitable method to solve the problem
                        scenmax.update()


                tScen = time.time() - tScen
                if tScen > MaxTime:
                    MaxTime = tScen
                if tScen < MinTime:
                    MinTime = tScen

            if time.time() - t0 < timeLimit:
                TimeMasterLP = TimeMasterLP + self.addBenders(self, method=BDmethod)
            self.PrimalMaster.update()
            tStart = time.time()
            self.PrimalMaster.optimize()
            TimeMasterLP = TimeMasterLP + (time.time() - tStart)
            PrimalLB = self.PrimalMaster.objval
            wrtStr = str(iter) + '\t' + str(time.time() - t0) + '\t' + str(PrimalLB) + '\t' + str(
                TimeMasterLP) + '\t' + str(TimeCutLP) + '\t' + str(TimeCutQP) + '\t' + str(TimeBaseMIP) + \
                     '\t' + str(TimeSub) + '\t' + str(AvgBaseSize) + '\t' + str(NscenSolved) + '\t' + str(
                MaxTime) + '\t' + str(MinTime) + '\t' + str(MaxSize) + '\t' + str(MinSize) + '\t' + \
                     str(BDmethod) + '\t' + str(self.Ncuts) + '\t' + str(self.Nsubs) + '\n'
            f = open("Results/" + str(self.name) + "_IterLag_" + str(gapTol) + "_" + str(pi0Coef) + ".txt", "a")
            f.write(wrtStr)
            f.close

    # Restricted separation of Lagrangian cuts
    def IterRstrLag(self, BDmethod='level', maxNcuts=10, pi0Coef=1e-2, tol=1e-4, gapTol=5e-1, timeLimit=60 * 60,
                    poolplus=False, LevelParam=0.01, mip=True, newn=True, earlyTmn=False):
        wrtStr = 'BendersMethod=' + str(BDmethod) + '\tmaxNcuts=' + str(maxNcuts) + '\ttol=' + str(
            tol) + '\tgapTol=' + str(gapTol) + '\tpi0Coef=' + str(pi0Coef) + \
                 '\ttimeLimit=' + str(timeLimit) + '\tpoolplus=' + str(poolplus) + '\tLevelParam=' + str(
            LevelParam) + '\tmip=' + str(mip) + '\tnewn=' + str(newn) + '\n'
        fileName = "Results/" + str(self.name) + "_IterRstrLag_" + str(gapTol) + "_" + str(maxNcuts) + "_" + str(
            newn) + "_" + str(pi0Coef) + "_" + str(LevelParam) + "_" + str(mip)
        if poolplus == True:
            fileName = fileName + "_pplus"
        if earlyTmn == True:
            fileName = "Results/" + str(self.name) + "_BnC"
        fileName = fileName + ".txt"
        f = open(fileName, "a")
        f.write(wrtStr)
        f.close
        t0 = time.time()
        TimeMasterLP = 0.0
        TimeCutLP = 0.0
        TimeCutQP = None
        TimeBaseMIP = 0.0
        TimeSub = 0.0
        x_value = {}                                # Master problem variables 
        theta_value = {}                            # Master problem variables - approximation
        iter = 0
        Iter_Bound = []
        Iter_Time = []
        CutHistory = {}
        for s in range(self.Nscen):
            CutHistory[s] = []

        TimeMasterLP = TimeMasterLP + self.addBenders(self, method=BDmethod)
        LPbnd = self.PrimalMaster.objval
        Iter_Bound.append(LPbnd)
        Iter_Time.append(time.time() - t0)
        # Initialize with perfect information solution
        piHat = {}
        subg = {}
        for i in range(self.Nfv):
            piHat[i] = self.cVec[i]
            subg[i] = -self.cVec[i]
        pi0Hat = 1.0
        for s in range(self.Nscen):
            tStart = time.time()
            tlimitValue = max(timeLimit - (time.time() - t0), 0)
            ObjV, xHat, yObjV, SubOpt, BestBound = self.SolveScenSub(scen_id=s, objCoef=piHat, regCoefy=pi0Hat,
                                                                     tlimit=tlimitValue)                            # solve a subproblem (12) with given dual (0,...0, 1)
            TimeSub = TimeSub + (time.time() - tStart)
            coef = [xHat[i] for i in range(self.Nfv)]
            coef.append(yObjV)
            CutHistory[s].append(coef)
            for soln in SubOpt:
                coef = [soln[i] for i in range(self.Nfv + 1)]
                CutHistory[s].append(coef)                                                                          # obtain the corresponding optimal x and y => z, theta, the coefficients for linear approximation of Q

        FindCut = True
        StopCondt = False
        # Add new stopping condt
        while FindCut == True and time.time() - t0 < timeLimit and StopCondt == False:
            iter += 1                                                                               # the number of iteration in the outest
            AvgBaseSize = 0
            NscenSolved = 0
            MaxTime = -float('inf')
            MinTime = float('inf')
            self.PrimalMaster.update()
            tStart = time.time()
            self.PrimalMaster.optimize()
            TimeMasterLP = TimeMasterLP + (time.time() - tStart)
            IterTime = time.time() - t0
            print('RstrLag. Iter ' + str(iter) + ', PrimalMaster LB: ' + str(
                self.PrimalMaster.objval) + ', Total Time: ' + str(IterTime) + ', Ncuts: ' + str(self.Ncuts))
            Iter_Time.append(IterTime)
            Iter_Bound.append(self.PrimalMaster.objval)
            FindCut = False
            x_value = {}
            theta_value = {}
            for j in range(self.Nfv):
                x_value[j] = self.x[j].x
            for s in range(self.Nscen):
                theta_value[s] = self.theta[s].x
            PredValList = []
            CoefBase = {}
            BaseSize = {}                                                                   # the Benders cut pool used to approximate Lag. Dual Multipliers
            ScenSet = []
            for s in range(self.Nscen):
                Ncuts = min(maxNcuts, len(self.cutlist[s]))
                RstrLagIter = 0

                if mip == False:                                                            
                    # 1. Choose last Benders' cuts
                    BaseSize[s] = Ncuts
                    CoefBase[s] = {}
                    for k in range(Ncuts):
                        CoefBase[s][k] = self.cutlist[s][-k - 1].copy()
                    ScenSet = range(self.Nscen)
                else:                                                                      
                    # 2. Approximate cut selection problem by MIP
                    # Increase the pool
                    if poolplus == True:
                        # Add unconstrained solution
                        scenmax = Model('scenmax')
                        piScenmax = {}                                                  # pi
                        absPiScenmax = {}                                               # |pi|
                        pi0 = scenmax.addVar(lb=0.0)                                    # pi0
                        lpi = scenmax.addVar(lb=-GRB.INFINITY)                          # tau - linear approximation of Q(pi, pi0)
                        for i in range(self.Nfv):
                            piScenmax[i] = scenmax.addVar(lb=-GRB.INFINITY)
                            absPiScenmax[i] = scenmax.addVar(lb=0.0)
                            scenmax.addConstr(piScenmax[i] <= absPiScenmax[i])
                            scenmax.addConstr(-piScenmax[i] <= absPiScenmax[i])
                        scenmax.addConstr(quicksum(absPiScenmax[i] for i in range(self.Nfv)) + pi0 <= 1)
                        # Add stored solutions
                        for coef in CutHistory[s]:
                            scenmax.addConstr(
                                lpi <= quicksum(coef[i] * piScenmax[i] for i in range(self.Nfv)) + coef[self.Nfv] * pi0)
                        scenmax.setObjective(
                                lpi - quicksum(piScenmax[i] * x_value[i] for i in range(self.Nfv)) - pi0 * theta_value[s])
                        scenmax.modelSense = GRB.MAXIMIZE
                        scenmax.setParam('OutputFlag', False)

                        coef = {}
                        scenmax.update()
                        scenmax.optimize()
                        objUB = scenmax.objval
                        subg_value = {}
                        subg_value[s] = {}
                        for i in range(self.Nfv):
                            subg_value[s][i] = sum(self.thetaCutList[s][k].pi * self.coeflist[s][k][i] for k in
                                                   range(len(self.thetaCutList[s]))) / self.pVec[s]
                        normsum = sum(abs(subg_value[s][i]) for i in range(self.Nfv)) + 1
                        scenmax.setObjective(quicksum(
                            ((piScenmax[i] - subg_value[s][i] / normsum) * (piScenmax[i] - subg_value[s][i] / normsum))
                            for i in range(self.Nfv)) + (pi0 - 1 / normsum) * (pi0 - 1 / normsum))
                        scenmax.addConstr(
                            lpi - quicksum(piScenmax[i] * x_value[i] for i in range(self.Nfv)) - pi0 * theta_value[s] >= LevelParam * objUB
                                            )
                        scenmax.modelSense = GRB.MINIMIZE
                        scenmax.params.Method = 0
                        scenmax.params.TimeLimit = 10.0
                        scenmax.update()
                        QPtime = time.time()
                        scenmax.optimize()
                        if scenmax.status == 2:
                            if pi0.x >= 1e-6:
                                for i in range(self.Nfv):
                                    coef[i] = piScenmax[i].x / pi0.x
                                self.cutlist[s].append(coef)
                        else:
                            print('Scen ' + str(s) + ' unconstrained problem not solved...: ' + str(scenmax.status))

                    # MIP formulation                           ---------------------                        # choose cuts by problem (28)
                    cutlistLen = len(self.cutlist[s])
                    Ncuts = min(maxNcuts, len(self.cutlist[s]))
                    scenmaxhat = Model('scenmaxhat')                                                         # problem (28)
                    pih = {}
                    cutlamh = {}                                                                             # beta in  (22)
                    absCutlamh = {}
                    zlamh = {}
                    pi0h = scenmaxhat.addVar(lb=0.0)
                    lpih = scenmaxhat.addVar(lb=-GRB.INFINITY)
                    for k in range(cutlistLen):
                        cutlamh[k] = scenmaxhat.addVar(lb=-GRB.INFINITY)
                        absCutlamh[k] = scenmaxhat.addVar(lb=0.0)
                        zlamh[k] = scenmaxhat.addVar(vtype=GRB.BINARY)
                        scenmaxhat.addConstr(cutlamh[k] <= absCutlamh[k])
                        scenmaxhat.addConstr(-cutlamh[k] <= absCutlamh[k])
                        scenmaxhat.addConstr(absCutlamh[k] <= zlamh[k])
                    for i in range(self.Nfv):
                        pih[i] = scenmaxhat.addVar(lb=-GRB.INFINITY)
                        scenmaxhat.addConstr(
                            pih[i] == quicksum(cutlamh[k] * self.cutlist[s][k][i] for k in range(cutlistLen)))   # constraint (22)
                    scenmaxhat.addConstr(quicksum(zlamh[k] for k in range(cutlistLen)) <= Ncuts)                 # constraint (25)
                    scenmaxhat.addConstr(quicksum(absCutlamh[k] for k in range(cutlistLen)) + pi0Coef * pi0h <= 1) # constraint in (20)

                    for coef in CutHistory[s]:
                        scenmaxhat.addConstr(
                            lpih <= quicksum(coef[i] * pih[i] for i in range(self.Nfv)) + coef[self.Nfv] * pi0h)

                    scenmaxhat.setObjective(
                        lpih - quicksum(pih[i] * x_value[i] for i in range(self.Nfv)) - pi0h * theta_value[s])
                    # until now, we just established problem (28) with normization condition in (20)

                    scenmaxhat.modelSense = GRB.MAXIMIZE
                    scenmaxhat.setParam('OutputFlag', False)

                    scenmaxhat.update()
                    tStart = time.time()
                    scenmaxhat.optimize()
                    TimeBaseMIP = TimeBaseMIP + (time.time() - tStart)
                    PredValList.append(scenmaxhat.objval)

                    BaseSize[s] = 0
                    CoefBase[s] = {}
                    for t in range(cutlistLen):
                        if zlamh[t].x > 0.5:                            
                            CoefBase[s][BaseSize[s]] = self.cutlist[s][t].copy()                                # CoefBase is the cuts chosen by MIP
                            BaseSize[s] = BaseSize[s] + 1
                            if t == cutlistLen - 1 and poolplus == True:
                                print('Scen ' + str(s) + " generated vector accepted")

                    if PredValList[s] <= tol * (abs(theta_value[s]) + 1):                                       # upper bound is too small, we terminate the alg
                        print('Scen ' + str(s) + ' predicted value <= tol')
                    else:
                        ScenSet.append(s)                                                                       # add this scenario into ScenSet

            if len(ScenSet) > 0:
                AvgBaseSize = sum(BaseSize[s] for s in ScenSet) / len(ScenSet)
                MaxSize = max(len(CutHistory[s]) for s in ScenSet)
                MinSize = min(len(CutHistory[s]) for s in ScenSet)

            for s in ScenSet:
                tScen = time.time()
                if time.time() - t0 > timeLimit:
                    break
                NscenSolved = NscenSolved + 1
                scenmax = Model('scenmax')
                piScenmax = {}
                cutlam = {}                                                                                         # beta in (22)
                for k in range(BaseSize[s]):
                    cutlam[k] = scenmax.addVar(lb=-GRB.INFINITY)
                pi0 = scenmax.addVar(lb=0.0)
                lpi = scenmax.addVar(lb=-GRB.INFINITY)
                piConstr = {}
                for i in range(self.Nfv):
                    piScenmax[i] = scenmax.addVar(lb=-GRB.INFINITY)
                    piConstr[i] = scenmax.addConstr(                                                                # constraint (22) for each component
                        piScenmax[i] == quicksum(cutlam[k] * CoefBase[s][k][i] for k in range(BaseSize[s]))         # pi is a linear combination of previous Benders cut
                                        )
                if newn == True:                                                                                    # control (19) or (20)
                    absCutlam = {}
                    for k in range(BaseSize[s]):                                            
                        absCutlam[k] = scenmax.addVar(lb=0.0)
                        scenmax.addConstr(cutlam[k] <= absCutlam[k])
                        scenmax.addConstr(-cutlam[k] <= absCutlam[k])
                    scenmax.addConstr(quicksum(absCutlam[k] for k in range(BaseSize[s])) + pi0Coef * pi0 <= 1)      # (20)
                else:
                    absPiScenmax = {}
                    for i in range(self.Nfv):
                        absPiScenmax[i] = scenmax.addVar(lb=0.0)
                        scenmax.addConstr(piScenmax[i] <= absPiScenmax[i])
                        scenmax.addConstr(-piScenmax[i] <= absPiScenmax[i])
                    scenmax.addConstr(quicksum(absPiScenmax[i] for i in range(self.Nfv)) + pi0Coef * pi0 <= 1)      # (19)

                scenmax.setObjective(
                    lpi - quicksum(piScenmax[i] * x_value[i] for i in range(self.Nfv)) - pi0 * theta_value[s])
                scenmax.modelSense = GRB.MAXIMIZE
                scenmax.setParam('OutputFlag', False)

                # Add stored solutions
                for coef in CutHistory[s]:
                    scenmax.addConstr(
                        lpi <= quicksum(coef[i] * piScenmax[i] for i in range(self.Nfv)) + coef[self.Nfv] * pi0)
                                                                                                                    # until now, we have established problem (28) but without binary variables to choose benders duals
                scenmax.update()
                tStart = time.time()
                scenmax.optimize()
                TimeCutLP = TimeCutLP + (time.time() - tStart)

                ContinueCondition = True
                RstrLagIter = 0
                piHat = {}
                for i in range(self.Nfv):
                    piHat[i] = piScenmax[i].x
                pi0Hat = pi0.x
                piBest = piHat
                pi0Best = pi0Hat
                lpiold = float('inf')
                LB = -float('inf')
                UB = float('inf')
                while ContinueCondition == True and time.time() - t0 < timeLimit:                               # this iteration is for find a good dual multiplier (pi, pi0)
                    RstrLagIter += 1                                                                            # the number of iterations to generate a Lag. cut
                    if UB < tol * (abs(theta_value[s]) + 1):                                                    # in alg.1, if UB is too small, we terminate the alg
                        print('RstrLag. Cut iter: ' + str(RstrLagIter) + ', scenario ' + str(s) + ': UB = ' + str(
                            UB) + ' < tol, base size: ' + str(BaseSize[s]))
                        break                                                                                   # if the UB is too small, we break, and do NOT add a Lag. cut
                    tStart = time.time()
                    tlimitValue = max(timeLimit - (time.time() - t0), 0)
                    ObjV, xHat, yObjV, SubOpt, BestBound = self.SolveScenSub(scen_id=s, objCoef=piHat, regCoefy=pi0Hat,
                                                                             tlimit=tlimitValue)
                    TimeSub = TimeSub + (time.time() - tStart)

                    for soln in SubOpt:
                        scenmax.addConstr(
                            lpi <= quicksum(soln[i] * piScenmax[i] for i in range(self.Nfv)) + soln[self.Nfv] * pi0)
                    scenmax.addConstr(lpi <= quicksum(xHat[i] * piScenmax[i] for i in range(self.Nfv)) + yObjV * pi0)

                    coef = [xHat[i] for i in range(self.Nfv)]
                    coef.append(yObjV)
                    CutHistory[s].append(coef)
                    for soln in SubOpt:
                        coef = [soln[i] for i in range(self.Nfv + 1)]
                        CutHistory[s].append(coef)
                    gap = BestBound - pi0Hat * theta_value[s] - sum(piHat[i] * x_value[i] for i in range(self.Nfv))
                    if gap > LB:
                        LB = gap
                        piBest = piHat.copy()
                        pi0Best = pi0Hat
                        lBest = BestBound

                    scenmax.update()
                    tStart = time.time()
                    scenmax.optimize()
                    TimeCutLP = TimeCutLP + (time.time() - tStart)
                    UB = scenmax.objval                                                                     # UB of problem (17)

                    if RstrLagIter % 100 == 0:
                        print('RstrLag. Cut iter: ' + str(RstrLagIter) + ', scenario: ' + str(s) + ', UB: ' + str(
                            UB) + ', LB: ' + str(LB) + ', pi0: ' + str(pi0Hat))
                    if UB - LB < gapTol * UB or UB - LB < 1e-6 or RstrLagIter > 1000:
                        if pi0Best > 1e-6:
                            print('RstrLag. Cut iter: ' + str(RstrLagIter) + ', scenario: ' + str(
                                s) + ', violation: ' + str(LB / pi0Best) + ', total time: ' + str(
                                time.time() - t0) + ', base size: ' + str(BaseSize[s]))
                        else:
                            print('RstrLag. Cut iter: ' + str(RstrLagIter) + ', scenario: ' + str(
                                s) + ', pi0Hat <= 1e-6, total time: ' + str(time.time() - t0) + ', base size: ' + str(
                                BaseSize[s]))
                        ContinueCondition = False
                        if pi0Best > 1e-6 and LB >= tol * (abs(theta_value[s]) + 1):
                            # iteration finish for this scenario, and we will add a cut to Master problem
                            self.thetaCutList[s].append(
                                self.PrimalMaster.addConstr(
                                    pi0Best * self.theta[s] >= -quicksum(piBest[i] * self.x[i] for i in range(self.Nfv)) + lBest
                                                        )
                                                    )
                            coef = {}
                            for i in range(self.Nfv):
                                coef[i] = -piBest[i]
                            coef[self.Nfv] = pi0Best
                            self.coeflist[s].append(coef)
                            self.Ncuts += 1
                            FindCut = True                                                                      # continue the outest iteration
                    else:
                        piHatOld = piHat.copy()
                        pi0HatOld = pi0Hat
                        for i in range(self.Nfv):
                            piHat[i] = piScenmax[i].x
                        pi0Hat = pi0.x
                        if max(abs(piHat[i] - piHatOld[i]) for i in range(self.Nfv)) < 1e-10 and abs(
                                pi0Hat - pi0HatOld) < 1e-10 and abs(lpi.x - lpiold) < 1e-10:
                            print('Same Solution!')
                            if pi0Best > 1e-6:
                                print('RstrLag. Cut iter: ' + str(RstrLagIter) + ', scenario: ' + str(
                                    s) + ', violation: ' + str(LB / pi0Best) + ', total time: ' + str(
                                    time.time() - t0) + ', base size: ' + str(BaseSize[s]))
                            ContinueCondition = False                                                               # we finish this Lag. cut
                            if pi0Best > 1e-6 and LB >= tol * (abs(theta_value[s]) + 1):                            # satisfy certain conditions, we then add a cut to Master problem
                                self.thetaCutList[s].append(self.PrimalMaster.addConstr(
                                    pi0Best * self.theta[s] >= -quicksum(
                                        piBest[i] * self.x[i] for i in range(self.Nfv)) + lBest))
                                # record the cut
                                coef = {}
                                for i in range(self.Nfv):
                                    coef[i] = -piBest[i]
                                coef[self.Nfv] = pi0Best
                                self.coeflist[s].append(coef)
                                self.Ncuts += 1                                                                     # record the cut number
                                FindCut = True                                                                      # continue the outest iteration
                        lpiold = lpi.x              
                tScen = time.time() - tScen
                if tScen > MaxTime:
                    MaxTime = tScen
                if tScen < MinTime:
                    MinTime = tScen

            if time.time() - t0 < timeLimit:
                TimeMasterLP = TimeMasterLP + self.addBenders(self, method=BDmethod)
            self.PrimalMaster.update()
            tStart = time.time()
            self.PrimalMaster.optimize()
            TimeMasterLP = TimeMasterLP + (time.time() - tStart)
            PrimalLB = self.PrimalMaster.objval
            Iter_Bound.append(self.PrimalMaster.objval)
            Iter_Time.append(time.time() - t0)
            if earlyTmn == True:
                if len(Iter_Bound) >= 6:
                    if Iter_Bound[-1] - Iter_Bound[-6] < 0.01 * (Iter_Bound[-1] - Iter_Bound[0]):
                        StopCondt = True

            wrtStr = str(iter) + '\t' + str(time.time() - t0) + '\t' + str(PrimalLB) + '\t' + str(
                TimeMasterLP) + '\t' + str(TimeCutLP) + '\t' + str(TimeCutQP) + '\t' + str(TimeBaseMIP) + \
                     '\t' + str(TimeSub) + '\t' + str(AvgBaseSize) + '\t' + str(NscenSolved) + '\t' + str(
                MaxTime) + '\t' + str(MinTime) + '\t' + str(MaxSize) + '\t' + str(MinSize) + '\t' + \
                     str(BDmethod) + '\t' + str(self.Ncuts) + '\t' + str(self.Nsubs) + '\n'
            f = open(fileName, "a")
            f.write(wrtStr)
            f.close
