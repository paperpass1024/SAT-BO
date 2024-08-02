import numpy as np
import os
import getData
import makeVirtual


class SAT:
    def __init__(self, inputPath, vitrualPath, rnname, solNum):
        self.data = getData.DATA()
        self.virtual = makeVirtual.virtual()
        self.varNum = 0
        self.solNum = solNum
        self.timesLimit = 10
        self.cnf = []
        self.clauseNum = 0
        self.inputPath = inputPath
        self.runName = rnname
        self.InputFileName = self.getFileName(inputPath)
        self.satRunFilePath = self.data.getRunPath(rnname)
        self.ubcsatRunFilePath = self.data.getRunPath("ubcsat")
        self.solvePath = self.data.getSolvePath(self.InputFileName)
        self.varWeightPath = self.data.getVarWerightPath(self.InputFileName)
        self.virtualFlowPath = vitrualPath
        self.virtualType = vitrualPath.split(".")[-1]
        self.logPath = self.data.getLogPath(self.InputFileName)
        self.pyLogPath = os.path.join(
            self.data.projectDir, "log/pyLog/" + self.getLogName(self.InputFileName)
        )
        # self.runSAT("randSample")
        self.getCnfInfo()

    def getCnfInfo(self):
        f = open(self.inputPath, "r")
        mess = f.readline().strip("\n").split(" ")
        self.varNum = int(mess[2])
        self.clauseNum = int(mess[3])
        for i in range(self.clauseNum):
            cnf = []
            line = f.readline().split(" 0")[0].split(" ")
            for j in line:
                cnf.append(int(j))
            self.cnf.append(cnf)
        f.close()
        self.runSAT("randSample")
        f = open(self.solvePath)
        self.solNum = int(f.readline().strip("\n").split(" ")[2])
        self.solNum = max(self.solNum, 3)
        f.close()
        # print("innitend")

    def getFileName(self, filePath):
        return filePath.split("\\")[-1].split("/")[-1]

    def getSolName(self, filename):
        return filename.split(".")[0] + ".sol"

    def getVWName(self, filename):
        return filename.split(".")[0] + ".vw"

    def getFlowName(self, filename):
        return filename.split(".")[0] + ".txt"

    def getLogName(self, filename):
        return filename.split(".")[0] + ".log"

    def runSAT(self, ctrlstr):
        config = ""
        if ctrlstr == "sat":
            config = self.sat()
        elif ctrlstr == "randSample":
            config = self.randSample()
        order = self.satRunFilePath + " " + config

        # print(order)
        os.system(order)

    def sat(self):
        config = (
            self.inputPath
            + " "
            + self.solvePath
            + " "
            + self.varWeightPath
            + " "
            + str(self.timesLimit)
            + " "
            + str(self.solNum)
        )
        return config

    def randSample(self):
        config = (
            self.inputPath
            + " "
            + self.solvePath
            + " "
            + str(self.timesLimit)
            + " "
            + str(self.solNum)
        )
        return config

    def getX(self):
        f = open(self.solvePath)
        info = f.readline().strip("\n").split(" ")
        solNum = int(info[2])
        X = [
            [int(x) > 0 for x in f.readline().split(" 0\n")[0].split(" ")]
            for i in range(solNum)
        ]
        # if solNum < self.solNum:
        #     for i in range(self.solNum - solNum):
        #         X.append(X[0])
        f.close()
        return np.array(X)

    def updataVarWeight(self, Weight):
        # print(len(Weight[0]))
        f = open(self.varWeightPath, "w")
        if self.runName == "walksat":
            id = 0
            for i in Weight[0]:
                id = id + 1
                if i < 0.5:
                    f.write(str("-"))
                f.write(str(id) + " ")
            f.write("\n")
        else:
            f.write(str(len(Weight)) + "\n")
            for idweight in Weight:
                for i in idweight:
                    f.write(str(i) + " ")
                f.write("\n")
        f.close()

    # X=[0 1 0 ....]

    def getFX(self, X):
        p = self.varNum
        sols = [[(i + 1) * (1 - 2 * (x[i] == 0)) for i in range(p)] for x in X]
        if self.virtualType == "cnf":
            FX = self.virtual.calSat(p, sols, self.virtualFlowPath)
        else:
            FX = self.virtual.cal(p, sols, self.virtualFlowPath)
        return np.array([[float(fx)] for fx in FX])

    def checkSolve(self, X):
        for sol in X:
            flag = 1
            for cnf in self.cnf:
                f = 0
                for i in cnf:
                    num = (int(sol[abs(i) - 1]) + int(i < 0)) % 2
                    f = f | num
                flag = flag & f
            if flag == 0:
                return False
        return True

    def getMax(self):
        if self.virtualType == "cnf":
            return self.virtual.getSatMax(self.varNum, self.virtualFlowPath)
        else:
            return self.virtual.getMax(self.varNum, self.virtualFlowPath)
