from turbo import Turbo1
import os
import setSAT
import sys
import argparse
import numpy as np
import setLevy
import time


def runTurbo(turbo, sat):
    # print("maxeval")
    # print(turbo.max_evals)
    # print("solnum")
    # print(sat.solNum)
    iter = 0
    while turbo.n_evals < turbo.max_evals - turbo.batch_size:
        # Generate and evalute initial design points
        # 3. （外部平台）将X修正为X'，获得X'对应的目标函数值fX'
        sat.runSAT("randSample")
        X_init = sat.getX()
        fX_init = sat.getFX(X_init)
        turbo.initial_solutions_of_trust_region(X_init, fX_init)
        # Thompson sample to get next suggestions
        while (
            turbo.n_evals < turbo.max_evals - turbo.batch_size
            and turbo.is_trust_region_big_enough()
        ):
            # 1. get suggest var assignment
            suggestSample = turbo.suggested_sampling()
            prefers = []
            for p in suggestSample:
                s = ""
                for i in np.around(p, 1):
                    s = s + str(i) + " "
                prefers.append(s)
            prefers = np.array(
                [
                    [float(i) for i in s.split(" ") if i != ""]
                    for s in list(set(prefers))
                ]
            )
            if len(prefers) > 2:
                prefers = np.array([prefers[i] for i in range(2)])
            # 2. get samples
            sat.updataVarWeight(prefers)
            # print("iter")
            sat.runSAT("sat")
            X_next = sat.getX()
            if len(X_next) == 0:
                timesLimit = sat.timesLimit
                for t in [20, 40, 60, 100]:
                    sat.timesLimit = t
                    X_next = sat.getX()
                    if len(X_next) != 0:
                        break
                sat.timesLimit = timesLimit
            if len(X_next) == 0:
                iter = iter + 1
                continue
            # while sat.checkSolve(X_next) is False:
            #     X_next = sat.getX()
            X_next = turbo.select_from_candidates(X_next)
            tmp = [[int(i) for i in x] for x in X_next]
            if sat.checkSolve(tmp) == False:
                print(tmp)
            # print(X_next)
            fX_next = sat.getFX(X_next)

            # Evaluate batch
            # 3. update
            turbo.solutions_and_obj_after_sampling(X_next, fX_next)
            iter = iter + 1

    # # print("jiaohu:"+str(kkk))


def OneFlowAns(turbo1, sat):
    runTurbo(turbo1, sat)
    x_best, f_best = turbo1.best_sol_and_obj()
    if len(x_best) == 0:
        return
    x_best = [int(i) for i in np.around(x_best, 0)]
    f_best = round(f_best[0], 4)
    MaxCover = float(sat.getMax())
    TrueCover = float(sat.getFX([x_best])[0][0])
    Coverage = TrueCover * 1.0 / MaxCover
    # Coverage = TrueCover
    info = str(
        "Best value found:\n\tf(x) = %.3f\nObserved at:\n\tx = %s\n"
        % (f_best, str(x_best).replace("\n", ""))
    )
    info = info + str("UpperLimitCover = %.4f\n" % (MaxCover))
    info = info + str("ActualCover = %.4f\n" % (TrueCover))
    info = info + str("Coverage = %.4f\n" % (Coverage))
    info = info + str("Is satisfied = %s\n" % (str(sat.checkSolve([x_best]))))
    info = info + str("Base Number = %f\n" % (sat.virtual.getBaseNum(sat.varNum)))
    return "\n>>>>>>ans info<<<<<<\n" + info


def getAnsWithoutBO(sat, times):
    info = ""
    info = (
        info
        + "\n>>>>>run infos<<<<<<\n"
        + "interaction = "
        + str(times)
        + "\n"
        + "varNum = "
        + str(sat.varNum)
        + "\n"
        + "clauseNum = "
        + str(sat.clauseNum)
        + "\n"
    )

    info = info + "\n>>>>>>ans info<<<<<<\n"
    prefers = [[0.5 for i in range(sat.varNum)]]
    sat.updataVarWeight(prefers)
    sat.runSAT("sat")
    X_next = sat.getX()
    fX_next = sat.getFX(X_next)
    x_best, f_best = 0, 0.0
    for i in range(len(X_next)):
        if fX_next[i][0] > f_best:
            f_best = fX_next[i][0]
            x_best = X_next[i]
    x_best = [int(i) for i in np.around(x_best, 0)]
    f_best = round(f_best, 4)
    MaxCover = float(sat.getMax())
    TrueCover = float(sat.getFX([x_best])[0][0])
    Coverage = TrueCover * 1.0 / MaxCover
    info = info + str(
        "Best value found:\n\tf(x) = %.3f\nObserved at:\n\tx = %s\n"
        % (f_best, str(x_best).replace("\n", ""))
    )
    info = info + str("UpperLimitCover = %.4f\n" % (MaxCover))
    info = info + str("ActualCover = %.4f\n" % (TrueCover))
    info = info + str("Coverage = %.4f\n" % (Coverage))
    info = info + str("Is satisfied = %s\n" % (str(sat.checkSolve([x_best]))))
    info = info + str("Base Number = %f\n" % (sat.virtual.getBaseNum(sat.varNum)))
    # print(info)
    return info


def optimize(current_directory, inputPath, sat, times):
    info = ""
    f = setLevy.Levy(sat.varNum)
    infos = (
        info
        + "\n>>>>>run infos<<<<<<\n"
        + "interaction = "
        + str(times)
        + "\n"
        + "varNum = "
        + str(sat.varNum)
        + "\n"
        + "clauseNum = "
        + str(sat.clauseNum)
        + "\n"
    )

    turbo1 = Turbo1(
        f=f,  # Handle to objective function
        lb=f.lb,  # Numpy array specifying lower bounds
        ub=f.ub,  # Numpy array specifying upper bounds
        # 解的数量
        n_init=sat.solNum,  # Number of initial bounds from an Latin hypercube design
        max_evals=(times - 1) * min(10, sat.solNum)
        + sat.solNum
        + 1,  # Maximum number of evaluations
        batch_size=min(10, sat.solNum),  # How large batch size TuRBO uses
        verbose=True,  # Print information from each batch
        use_ard=True,  # Set to true if you want to use ARD for the GP kernel
        max_cholesky_size=2000,  # When we switch from Cholesky to Lanczos
        n_training_steps=50,  # Number of steps of ADAM to learn the hypers
        min_cuda=1024,  # Run on the CPU for small datasets
        device="cpu",  # "cpu" or "cuda"
        dtype="float64",  # float64 or float32
    )
    infos = infos + OneFlowAns(turbo1, sat)
    return infos


if __name__ == "__main__":
    start_time = time.time()

    parser = argparse.ArgumentParser(description="argparse testing")
    parser.add_argument("--input", "-i", type=str, required=True)
    parser.add_argument("--traffic", "-t", type=str, required=True)
    parser.add_argument("--runName", "-r", type=str, default="sat")
    parser.add_argument(
        "--iteration", "--iteration", type=int, default=6, help="iteration"
    )
    parser.add_argument("--solveUppers", "-s", type=int, default=30)
    parser.add_argument("--isUseBo", "--isUseBo", type=int, default=1)
    args = parser.parse_args()
    current_directory = os.path.dirname(os.path.abspath(__file__))

    # ========================================================
    if args.isUseBo == 1:
        sat = setSAT.SAT(args.input, args.traffic, args.runName, args.solveUppers)
        # sys.stdout = open(sat.pyLogPath, "w")
        ansInfo = optimize(current_directory, args.input, sat, int(args.iteration))
    else:
        sat = setSAT.SAT(
            args.input,
            args.traffic,
            args.runName,
            args.solveUppers * int(args.iteration),
        )
        sat.timesLimit = sat.timesLimit * int(args.iteration)
        # sys.stdout = open(sat.pyLogPath, "w")
        ansInfo = getAnsWithoutBO(sat, int(args.iteration))

    # ========================================================
    ansDir = os.path.abspath(os.path.join(current_directory, "../ans"))
    f = open(sat.logPath)
    info = "======== details ========"
    info = info + ansInfo
    end_time = time.time()
    execution_time = end_time - start_time
    times = "time = %.4f\n" % (execution_time)
    info = info + times
    ansPath = os.path.join(ansDir, sat.InputFileName.split(".cnf")[0] + ".ans")
    f = open(ansPath, "w")
    f.write(info)
    f.close()
    print(info)
    #
