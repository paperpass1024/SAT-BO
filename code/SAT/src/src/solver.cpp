
#include "solver.h"
// =====================================================================================
namespace SAT{
    bool Solver::check(VariableValue &X, SATProblem &SAT) {
        for (auto cln : SAT.C) {
            bool flag = 0;
            for (auto i : cln) {
                int id = abs(i);
                flag |= (i < 0 ? (1 ^ X[id]) : X[id]);
            }
            if (!flag) return false;
        }
        return true;
    }

    bool Solver::check(vector<VariableValue> &Xs, SATProblem &SAT) {
        for (auto X : Xs) {
            if(!check(X,SAT)) return false;
        }
        return true;
    }

}
