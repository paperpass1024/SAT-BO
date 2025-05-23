import getData
import math
import os
import random


class virtual:
    def __init__(self):
        self.data = getData.DATA()
        self.dataDir = os.path.join(self.data.projectDir, self.data.dataDirName)
        self.sat_k = [3.8, 3.9, 4.0, 4.1, 4.2]

    def sat(self):
        k = self.sat_k
        clauseLine = 3
        satPath = os.path.join(self.data.virtualDir, "sat")
        files = self.data.getFile(self.dataDir, ["*.cnf"])
        for file in files:
            fileName = self.data.getFileName(file)
            infos = fileName.split(".cnf")[0].split("-")
            p = int(infos[0].strip("v"))
            for case in range(5):
                path = os.path.join(satPath, str(case + 1))
                flowFile = os.path.join(path, fileName)
                cnf = int(p * k[case])
                lines = "p cnf " + str(p) + " " + str(cnf) + "\n"
                dic = {}
                i = 0
                while i < cnf:
                    tmp = []
                    line = ""
                    for j in range(clauseLine):
                        num = random.randint(1, p)
                        if random.randint(0, 1) == 1:
                            num = -num
                        tmp.append(num)
                    tmp = sorted(tmp, key=abs)
                    for j in tmp:
                        line = line + str(j) + " "
                    line = line + "0\n"
                    if line in dic.keys():
                        continue
                    i = i + 1
                    dic[line] = True
                    lines = lines + line
                f = open(flowFile, "w")
                f.write(lines)
                f.close()

    def binomial(self):
        binomialPath = os.path.join(self.data.virtualDir, "binomial")
        files = self.data.getFile(self.dataDir, ["*.cnf"])
        k = [5, 10, 15, 20, 25]
        for file in files:
            fileName = self.data.getFileName(file)
            infos = fileName.split(".cnf")[0].split("-")
            p = int(infos[0].strip("v"))
            for case in range(5):
                path = os.path.join(binomialPath, str(case + 1))
                flowFile = os.path.join(path, fileName.split(".cnf")[0] + ".txt")
                # if os.path.isfile(flowFile):
                #     continue
                point = []
                num = p * 2
                for i in range(num):
                    if random.randint(1, 100) <= k[case]:
                        point.append(random.randint(11, 100))
                    else:
                        point.append(random.randint(0, 10))
                f = open(flowFile, "w")
                for i in range(num):
                    f.write(str(point[i]))
                    f.write(" ")
                    if i == p - 1:
                        f.write("0 ")
                f.write("\n")
                f.close()

    def powerLaw(self):
        powerLawPath = os.path.join(self.data.virtualDir, "power-law")
        files = self.data.getFile(self.dataDir, ["*.cnf"])
        k = [2, 1, 3, 4, 5]
        for file in files:
            fileName = self.data.getFileName(file)
            infos = fileName.split(".cnf")[0].split("-")
            p = int(infos[0].strip("v"))
            for case in range(5):
                path = os.path.join(powerLawPath, str(case + 1))
                flowFile = os.path.join(path, fileName.split(".cnf")[0] + ".txt")
                if p == 0:
                    continue
                # if os.path.isfile(flowFile):
                #     continue
                point = []
                num = p * 2
                e = k[case]
                high = 1
                if e == 1:
                    high = max(high, int(math.log(num)))
                else:
                    high = max(high, int(math.log(num, e)))
                for i in range(high):
                    point.append(random.randint(1, 10) + 1000)
                for i in range(num - high):
                    point.append(random.randint(1, 10))
                f = open(flowFile, "w")
                for i in range(num):
                    f.write(str(point[i]))
                    f.write(" ")
                    if i == p - 1:
                        f.write("0 ")
                f.write("\n")
                f.close()

    def halfNormal(self):
        normalPath = os.path.join(self.data.virtualDir, "half-normal")
        files = self.data.getFile(self.dataDir, ["*.cnf"])
        sigma = [30, 45, 60, 75, 90]
        mu = 0
        for file in files:
            fileName = self.data.getFileName(file)
            infos = fileName.split(".cnf")[0].split("-")
            p = int(infos[0].strip("v"))
            for case in range(5):
                path = os.path.join(normalPath, str(case + 1))
                flowFile = os.path.join(path, fileName.split(".cnf")[0] + ".txt")
                # if os.path.isfile(flowFile):
                #     continue
                point = []
                num = p * 2
                for i in range(num):
                    point.append(abs(int(random.normalvariate(mu, sigma[case]))))
                f = open(flowFile, "w")

                for i in range(num):
                    f.write(str(point[i]))
                    f.write(" ")
                    if i == p - 1:
                        f.write("0 ")
                f.write("\n")
                f.close()

    def random(self):
        randomPath = os.path.join(self.data.virtualDir, "random")
        files = self.data.getFile(self.dataDir, ["*.cnf"])
        for file in files:
            fileName = self.data.getFileName(file)
            infos = fileName.split(".cnf")[0].split("-")
            p = int(infos[0].strip("v"))
            for case in range(5):
                path = os.path.join(randomPath, str(case + 1))
                flowFile = os.path.join(path, fileName.split(".cnf")[0] + ".txt")
                # if os.path.isfile(flowFile):
                #     continue
                point = []
                num = p * 2
                for i in range(num):
                    point.append(random.randint(1, 100))
                f = open(flowFile, "w")
                for i in range(num):
                    f.write(str(point[i]))
                    f.write(" ")
                    if i == p - 1:
                        f.write("0 ")
                f.write("\n")
                f.close()

    def getId(self, p, liter):
        return p + liter

    def getNum(self, num):
        return num * num - 1

    def cal(self, p, sols, virtualPath):
        ans = []
        f = open(virtualPath)
        tt = f.readline().strip(" \n").split(" ")
        f.close()
        flow = [int(tt[x]) for x in range(len(tt))]
        mx = self.getMaxTmp(p, virtualPath)
        for sol in sols:
            a = 0
            for j in sol:
                a = a + self.getNum(flow[self.getId(p, j)])
            ans_sol = a
            # ans_sol = self.changeAns(ans_sol * 100.0 / mx, p)
            ans.append(ans_sol)

        return ans

    def getMaxTmp(self, p, virtualPath):
        f = open(virtualPath)
        tt = f.readline().strip(" \n").split(" ")
        f.close()
        flow = [int(tt[x]) for x in range(len(tt))]
        ans = 0
        for i in range(p):
            num = max(flow[self.getId(p, i + 1)], flow[self.getId(p, -i - 1)])
            ans = ans + self.getNum(num)
        return ans

    def getMax(self, p, virtualPath):
        # return self.changeAns(100.0, p)
        f = open(virtualPath)
        tt = f.readline().strip(" \n").split(" ")
        f.close()
        flow = [int(tt[x]) for x in range(len(tt))]
        ans = 0
        for i in range(p):
            num = max(flow[self.getId(p, i + 1)], flow[self.getId(p, -i - 1)])
            ans = ans + self.getNum(num)
        return ans

    def getBaseNum(self, p):
        baseNum = 1.051 - p / 1000000.0
        baseNum = round(baseNum, 6)

        return baseNum

    def changeAns(self, sol, p):
        return math.pow(self.data.k, sol)

    def calSat(self, p, sols, virtualPath):
        ans = []
        f = open(virtualPath)
        cnf = int(f.readline().strip("\n").split(" ")[3])
        lines = [f.readline().strip(" 0\n").split(" ") for i in range(cnf)]
        cln = [[int(x) for x in lines[i]] for i in range(cnf)]
        f.close()
        for sol in sols:
            ans_sol = 0
            for clause in cln:
                flag = 0
                for x in clause:
                    flag = flag | (((sol[abs(x) - 1] > 0) + int(x < 0)) % 2)
                ans_sol = ans_sol + flag
            # print(ans_sol*100.0/cnf)
            ans_sol = self.changeAns(ans_sol * 100.0 / cnf, p)
            # print(ans_sol)
            ans.append(ans_sol)
        return ans

    def getSatMax(self, p, virtualPath):
        # f = open(virtualPath)
        # cnf = int(f.readline().strip("\n").split(" ")[3])
        # f.close()
        return self.changeAns(100.0, p)
