import os
import glob
import inspect


current_path = inspect.getfile(inspect.currentframe())
projectDir = os.path.abspath(os.path.join(os.path.dirname(current_path), "../"))
# print(projectDir)

class DATA:
    def __init__(self):
        self.k = 1.05
        self.conditions = []
        self.dataDirName = "benchmark_verificationRule-SAT-Encoding"
        self.types = ["random", "sat", "binomial", "power-law", "half-normal"]

        self.projectDir = projectDir
        self.dataDir = os.path.abspath(os.path.join(projectDir, self.dataDirName))
        self.virtualDir = os.path.abspath(os.path.join(projectDir, "virtual"))
        self.mainFile = os.path.abspath(os.path.join(projectDir, "code/main.py"))
        self.solveDir = os.path.abspath(os.path.join(projectDir, "log/solve"))
        self.ansDir = os.path.abspath(os.path.join(projectDir, "log/ans"))
        self.logDir = os.path.abspath(os.path.join(projectDir, "log/log"))
        self.weightDir = os.path.abspath(os.path.join(projectDir, "log/weight"))
        self.runFileDir = os.path.abspath(os.path.join(projectDir, "code/SAT/build"))

    def getFile(self, path, condition):
        g = os.walk(path)

        pathfile = []
        for path, dir_list, file_list in g:
            for condi in condition:
                for file_name in glob.glob(os.path.join(path, condi)):
                    pathfile.append(file_name)
        return pathfile

    def getFileName(self, file):
        return file.split("/")[-1]

    def sortData(self, files):
        tmp = []

        for file in files:
            fileName = self.getFileName(file)
            types = fileName.split("-")
            p = int(types[0].strip("v"))
            cnf = int(types[1].strip("c"))
            id = int(types[2].split(".cnf")[0])
            dic = {"name": file, "p": p, "cnf": cnf, "id": id}
            tmp.append(dic)

        tmp = sorted(tmp, key=lambda x: (x["p"], x["cnf"], x["id"]))
        files = []
        for file in tmp:
            files.append(file["name"])
        return files

    def select(self, p_low, cnf_low, files):
        files_select = []
        for file in files:
            fileName = self.getFileName(file)
            types = fileName.split("-")
            p = int(types[0].strip("v"))
            cnf = int(types[1].strip("c"))
            if p < p_low or cnf < cnf_low:
                continue
            files_select.append(file)
        return files_select

    def selectUpper(self, p_up, cnf_up, files):
        files_select = []
        for file in files:
            fileName = self.getFileName(file)
            types = fileName.split("-")
            p = int(types[0].strip("v"))
            cnf = int(types[1].strip("c"))
            if p > p_up or cnf > cnf_up:
                continue
            files_select.append(file)
        return files_select

    def getSolvePath(self, fileName):
        return os.path.join(self.solveDir, fileName.split(".cnf")[0] + ".sol")

    def getVirtualPath(self, typ, case, fileName):
        relativePath = os.path.join(typ, str(case))
        virtualDir = os.path.join(self.virtualDir, relativePath)
        virtualPath = os.path.join(virtualDir, fileName.split(".cnf")[0])
        if typ == "sat":
            virtualPath = virtualPath + ".cnf"
        else:
            virtualPath = virtualPath + ".txt"
        return virtualPath

    def getVarWerightPath(self, fileName):
        return os.path.join(self.weightDir, fileName.split(".cnf")[0] + ".vw")

    def getLogPath(self, fileName):
        return os.path.join(self.logDir, fileName.split(".cnf")[0] + ".log")

    def getRunPath(self, runName):
        return os.path.join(self.runFileDir, runName)

    def push(self):
        os.system("cd " + self.projectDir + " && ./.autopush.sh")

    def getHeadOfAns(self):
        head = [
            "fileName",
            "cnf Solver",
            "virtual type",
            "virtual case",
            "isUseBo",
            "interaction",
            "varNum",
            "clauseNum",
            "UpperLimitCover",
            "ActualCover",
            "Coverage",
            "Is satisfied",
            "Base Number",
            "time",
        ]
        return head


class file:
    def __init__(self, inputfile, data, typ, case, rnName, iterations, solUp, isUseBo):
        self.head = data.getHeadOfAns()
        self.typ = typ
        self.case = case
        self.runName = rnName
        self.inputFile = inputfile
        self.iterations = iterations
        self.isUseBo = isUseBo
        self.solUpper = solUp
        self.fileName = data.getFileName(inputfile)
        self.virtualPath = data.getVirtualPath(typ, case, self.fileName)
        self.solvePath = data.getSolvePath(self.fileName)
        self.weightPath = data.getVarWerightPath(self.fileName)
        self.logPath = data.getLogPath(self.fileName)
        self.runPath = data.getRunPath(rnName)
        self.runDir = data.runFileDir
        self.mainFile = data.mainFile
        self.pyLogPath = os.path.join(
            data.projectDir, "log/pyLog/" + self.fileName.split(".cnf")[0] + ".log"
        )
        self.ansPath = os.path.join(
            data.projectDir, "ans/" + self.fileName.split(".cnf")[0] + ".ans"
        )

    def getInfoOfSatBoAns(self):
        f = open(self.ansPath)
        dic = {}
        dic["fileName"] = self.ansPath.split("/")[-1].strip(".ans")
        dic["cnf Solver"] = self.runName
        dic["virtual type"] = self.typ
        dic["virtual case"] = self.typ + "-" + str(self.case)
        dic["isUseBo"] = bool(self.isUseBo)
        while True:
            line = f.readline()
            if line == "":
                break
            infos = line.strip("\n").split(" = ")
            if infos[0] in self.head:
                dic[infos[0]] = infos[1]
        f.close()
        return dic

    def SAT_BO(self):
        if os.path.isfile(self.ansPath):
            os.system("rm " + self.ansPath)
        # if os.path.isfile(self.virtualPath) is False:
        #     if self.typ == "sat":
        #         makeVirtual.virtual().sat()
        #     elif self.typ == "binomial":
        #         makeVirtual.virtual().binomial()
        #     elif self.typ == "power-law":
        #         makeVirtual.virtual().powerLaw()
        #     elif self.typ == "half-normal":
        #         makeVirtual.virtual().halfNormal()

        order = (
            "python -u "
            + self.mainFile
            + " -i "
            + self.inputFile
            + " -t "
            + self.virtualPath
            + " -r "
            + self.runName
            + " --iteration "
            + str(self.iterations)
            + " -s "
            + str(self.solUpper)
            + " --isUseBo "
            + str(self.isUseBo)
        )
        os.system(order)
        # print(order)
        f = open(self.ansPath)
        while True:
            lin = f.readline()
            if lin == "":
                print(order)
                break
            line = lin.strip("\n").split(" = ")
            if line[0] != "Is satisfied":
                continue
            flag = line[1]
            if flag != "True":
                self.SAT_BO()
            break
        f.close()

    def checkSATwithLog(self, timeLimit, solNum):
        if os.path.isfile(self.logPath):
            os.system("rm " + self.logPath)
        # os.system("cd " + self.runDir + " && make")
        if os.path.isfile(self.weightPath) is False:
            f = open(self.inputFile)
            p = int(f.readline().strip("\n").split(" ")[3])
            f.close()
            f = open(self.weightPath, "w")
            f.write(str(1) + "\n")
            for i in range(p):
                f.write(str(0) + " ")
            f.write("\n")
            f.close()
            print("creat")
        if self.isUseBo is False:
            timeLimit=int(timeLimit)*10
        order = (
            self.runPath
            + " "
            + self.inputFile
            + " "
            + self.solvePath
            + " "
            + self.weightPath
            + " "
            + str(timeLimit)
            + " "
            + str(solNum)
            + " 2> "
            + self.logPath
        )
        # print(order)
        os.system(order)
        f = open(self.logPath)
        dic = {}
        while True:
            lin = f.readline()
            if lin == "":
                print(order)
                break
            line = lin.strip("\n").split(" = ")
            if len(line) >= 2:
                dic[line[0]] = line[1]
        f.close()

    def checkSAT(self, timeLimit, solNum):
        # os.system("cd " + self.runDir + " && make")
        if os.path.isfile(self.weightPath) is False:
            f = open(self.inputFile)
            p = int(f.readline().strip("\n").split(" ")[3])
            f.close()
            f = open(self.weightPath, "w")
            f.write(str(1) + "\n")
            for i in range(p):
                f.write(str(0) + " ")
            f.write("\n")
            f.close()
            print("creat")
        order = (
            self.runPath
            + " "
            + self.inputFile
            + " "
            + self.solvePath
            + " "
            + self.weightPath
            + " "
            + str(timeLimit)
            + " "
            + str(solNum)
        )
        # print(order)
        os.system(order)

    def checkSATrandom(self, timeLimit, solNum):
        if os.path.isfile(self.logPath):
            os.system("rm " + self.logPath)
        order = (
            self.runPath
            + " "
            + self.inputFile
            + " "
            + self.solvePath
            + " "
            + str(timeLimit)
            + " "
            + str(solNum)
            + " 2> "
            + self.logPath
        )
        # print(order)
        os.system(order)
        f = open(self.logPath)

        dic = {}
        while True:
            lin = f.readline()
            if lin == "":
                break
            line = lin.strip("\n").split(" = ")
            if len(line) >= 2:
                dic[line[0]] = line[1]
        f.close()
        return dic

    def checkWalkSATrandom(self, timeLimit, solNum):
        if os.path.isfile(self.logPath):
            os.system("rm " + self.logPath)
        order = (
            self.runPath
            + " "
            + self.inputFile
            + " "
            + self.solvePath
            + " "
            + str(timeLimit)
            + " "
            + str(solNum)
            # + " 1> 1.txt "
            # + " 2> "
            # + self.logPath
        )
        # print(order)
        os.system(order)
        f = open(self.logPath)

        dic = {}
        while True:
            lin = f.readline()
            if lin == "":
                break
            line = lin.strip("\n").split(" = ")
            if len(line) >= 2:
                dic[line[0]] = line[1]
        f.close()
        return dic
