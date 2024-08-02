#include "sol.h"
#include <map>
#include <string>
namespace SAT{
class Solver {
    mt19937 pseudoRandNumGen;
    void initRand(int seed) { pseudoRandNumGen = mt19937(seed); }
    int fastRand(int lb, int ub) { return (pseudoRandNumGen() % (ub - lb)) + lb; }
    int fastRand(int ub) { return pseudoRandNumGen() % ub; }
    int rand(int lb, int ub) {
        return uniform_int_distribution<int>(lb, ub - 1)(pseudoRandNumGen);
    }
    int rand(int ub) {
        return uniform_int_distribution<int>(0, ub - 1)(pseudoRandNumGen);
    }

    long unsigned int outputs = 0;
    int limit_l = 0, limit, haspre;
    long unsigned int preLevel = 0;
    sol ansers[3];
    std::map<string,bool> mp;
    bool check(VariableValue &X, SATProblem &SAT);
    bool check(vector<VariableValue> &Xs, SATProblem &SAT);
    // ----------dpll
    void randomForIndependenceVar(sol &ans);
    bool isInvert(int x, sol &ans);
    int getlevel(sol &ans, int num, int allnum);
    bool dpll(sol &ans, vector<int> q, vector<VariableValue> &output,
    std::function<bool()> isTimeout);
    void randomForIndependenceVar();
    bool dpllForAssignment(sol &ans, vector<int> q, vector<VariableValue> &output, std::function<bool()> isTimeout);


public:
    bool solve(vector<VariableValue> &output, SATProblem &input,
     std::function<bool()> isTimeout, int seed,
     bool isHasAssignmentPrefer);
};
}
