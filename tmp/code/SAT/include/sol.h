#include <algorithm>
#include <cstdlib>
#include <random>

#include "SAT.h"
#include "node.h"
using namespace std;

namespace SAT {

class sol {
public:
    vector<bool> pos, ans;
    vector<VariableId> vis;
    vector<ClauseNode> F;
    vector<Node> X;
    vector<VariableId> X_id;
    vector<vector<ClauseId>> bucketOfFLen;
    vector<ClauseId> bucketId;
    ClauseId F_len = 0;
    VariableId X_len = 0;
    VariableId n, MinLen;
    ClauseId cln;
    vector<LiteralId> isSearch;
    bool isHasEmptyClause = false;
    int T = 10, T_MAX = 100000;
    const int basicLen = 2;
    inline void insert(VariableId &len, ClauseId clid);
    inline void remove(VariableId &len, ClauseId clid);
    inline int getIterId(LiteralId x);
    inline bool isSatisfied();
    inline void addQ(LiteralId x, vector<LiteralId> &decisionLevel);
    inline void InitialSpaceAllocation(SATProblem &input);
    inline bool init(SATProblem &input, std::vector<double> &assignmentPrefer);
    inline bool init(SATProblem &input);
    inline void vitrualRemoveNode(LiteralId liter);
    inline void vitrualAddNode();
    inline Node &getNode(LiteralId &x);
    inline void getAns(vector<bool> &output);
    inline void removeSatisfiedClauses(Clause &S, LiteralId &liter, vector<LiteralId> &q, vector<ClauseId> &changeClauseSet);
    inline void removeUnsatisfiedClauses(Clause &S, LiteralId liter, vector<LiteralId> &q, vector<ClauseId> &changeClauseSet);
    inline void addSatisfiedClauses(Clause &S, LiteralId &liter);
    inline void addUnsatisfiedClauses(Clause &S);
    bool unit_propagation(vector<LiteralId> &decisionLevel, vector<ClauseId> &changeClauseSet);
    void recovery(vector<LiteralId> &decisionLevel, vector<ClauseId> &changeClauseSet);
    bool single_unit_propagation(vector<LiteralId> &decisionLevel, vector<ClauseId> &changeClauseSet);
    void singleRecovery(vector<LiteralId> &decisionLevel, vector<ClauseId> &changeClauseSet);
    int getPreferAssignment(vector<VariableId> &can);
    void getCandicate(vector<VariableId> &can);
    vector<LiteralId> decisionLevl;
};
inline void sol::insert(VariableId &len, ClauseId clid) {
    bucketId[clid] = bucketOfFLen[len].size();
    bucketOfFLen[len].push_back(clid);
}
inline void sol::remove(VariableId &len, ClauseId clid) {
    vector<int> &bucket = bucketOfFLen[len];
    int clidEnd = bucket.back();
    bucketId[clidEnd] = bucketId[clid];
    bucket[bucketId[clid]] = clidEnd;
    bucket.pop_back();
}
inline int sol::getIterId(LiteralId x) {
    return (x ^ (x >> 31)) - (x >> 31);
}
inline bool sol::isSatisfied() { return (F_len == 0); }
inline void sol::addQ(LiteralId x, vector<LiteralId> &decisionLevel) {
    VariableId id = getIterId(x);
    if (pos[id]) return;
    pos[id] = true;
    decisionLevel.push_back(x);
}
inline void sol::InitialSpaceAllocation(SATProblem &input) {
    cln = input.clauseNum;  // 子句数量
    n = input.num;          // 变元数量
    if (cln * 1.0 / n < 3)
        T_MAX = 600;
    else
        T_MAX = 300;  // 单元子句传播上限
    ans.resize(n + 1);
    pos.resize(n + 1, 0), vis.resize(cln, 0);
    X_id.resize(n + 1), X.resize(n + 1);
    bucketOfFLen.resize(n + 1), F.resize(cln), bucketId.resize(cln);
    X_len = n;
    F_len = 0;
    isSearch.resize(n + 1, 0);
    for (int i = 0; i < n; i++) {
        X_id[i] = i + 1;
        X[i + 1].init(cln, i);
    }
}
inline bool sol::init(SATProblem &input, std::vector<double> &assignmentPrefer) {
    InitialSpaceAllocation(input);
    srand(time(NULL));
    vector<int> decisionLevel;
    for (int i = 0; i < cln; i++) {
        Clause clause = input.C[i];
        ClauseId clid = F_len;
        ClauseNode &cl = F[clid];
        if (!cl.createClause(clause, n)) continue;
        if (cl.len == 1) addQ(cl.C[1], decisionLevel);
        insert(cl.len, clid);
        for (int j = 1; j <= cl.len; j++) {
            int &liter = cl.C[j];
            getNode(liter).addS(clid, (liter > 0));
        }
        F_len++;
    }
    /* for(int i=0;i<X_len;i++){ */
    /* 	/1* int x=X_id[i]; *1/ */
    /* 	/1* Node &node=X[x]; *1/ */

    /* } */
    if (assignmentPrefer.size()) {
        for (int i = 0; i < n; i++) {
            X[i + 1].assignmentPrefer = assignmentPrefer[i] * 10;
            ans[i] = 0;
        }
    }
    else{
        for (int i = 0; i < n; i++) {
            X[i + 1].assignmentPrefer = 5;
            ans[i] = 0;
        }


    }
    vector<int> tmp;
    if (!unit_propagation(decisionLevel, tmp)) return false;
    decisionLevel.clear();

    // for(int i=0;i<X_len;i++) if(X[X_id[i]].isEmpty())
    // decisionLevel.push_back(X_id[i]); for(auto i:decisionLevel)
    // vitrualRemoveNode(i);
    return true;
}
inline bool sol::init(SATProblem &input) {
    InitialSpaceAllocation(input);
    vector<int> decisionLevel;
    for (int i = 0; i < cln; i++) {
        Clause clause = input.C[i];
        ClauseId clid = F_len;
        ClauseNode &cl = F[clid];
        if (!cl.createClause(clause, n)) continue;
        if (cl.len == 1) addQ(cl.C[1], decisionLevel);
        insert(cl.len, clid);
        for (int j = 1; j <= cl.len; j++) {
            int &liter = cl.C[j];
            getNode(liter).addS(clid, (liter > 0));
        }
        F_len++;
    }
    /* for(int i=0;i<X_len;i++){ */
    /* 	int x=X_id[i]; */
    /* 	/1* Node &node=X[x]; *1/ */

    /* } */
    vector<int> tmp;
    if (!unit_propagation(decisionLevel, tmp)) return false;
    decisionLevel.clear();

    // for(int i=0;i<X_len;i++) if(X[X_id[i]].isEmpty())
    // decisionLevel.push_back(X_id[i]); for(auto i:decisionLevel)
    // vitrualRemoveNode(i);
    return true;
}
inline void sol::vitrualRemoveNode(LiteralId liter) {
    liter = getIterId(liter);
    X_len--;
    LiteralId en = X_id[X_len];
    swap(X_id[X[liter].name], X_id[X_len]);
    swap(X[liter].name, X[en].name);
}
inline void sol::vitrualAddNode() { X_len++; }
inline Node &sol::getNode(LiteralId &x) { return X[getIterId(x)]; }
inline void sol::getAns(vector<bool> &output) { output = ans; }
inline void sol::removeSatisfiedClauses(Clause &S, LiteralId &liter, vector<LiteralId> &q, vector<ClauseId> &changeClauseSet) {
    for (int i = 1; i <= S[0]; i++) {
        ClauseId &clause = S[i];
        ClauseNode &cl = F[clause];
        if (!vis[clause]) {
            vis[clause] = cl.len * -1;
            changeClauseSet.push_back(clause);
        } else if (vis[clause] > 0)
            vis[clause] *= -1;
        for (int j = 1; j <= cl.len; j++) {
            LiteralId x = cl.C[j];
            if (x == liter) continue;
            Node &node = getNode(x);
            node.virtualRemoveS(clause, (x > 0));
        }
    }
    F_len -= S[0];
}
inline void sol::removeUnsatisfiedClauses(Clause &S, LiteralId liter, vector<LiteralId> &q, vector<ClauseId> &changeClauseSet) {
    for (int i = 1; i <= S[0]; i++) {
        ClauseNode &cl = F[S[i]];
        if (!vis[S[i]]) {
            vis[S[i]] = cl.len;
            changeClauseSet.push_back(S[i]);
        }
        cl.virtualRemoveLiter(liter);
        if (cl.len == 0) isHasEmptyClause = true;
        if (cl.len == 1 && !pos[getIterId(cl.C[1])]) {
            q.push_back(cl.C[1]);
            pos[getIterId(cl.C[1])] = 1;
        }
    }
}
inline void sol::addSatisfiedClauses(Clause &S, LiteralId &liter) {
    F_len += S[0];
    for (int i = S[0]; i >= 1; i--) {
        ClauseId &clause = S[i];
        ClauseNode &cl = F[clause];
        for (int j = cl.len; j >= 1; j--) {
            LiteralId &x = cl.C[j];
            if (x == liter) continue;
            getNode(x).virtualAddS(x);
        }
    }
}
inline void sol::addUnsatisfiedClauses(Clause &S) {
    for (int i = S[0]; i >= 1; i--) {
        ClauseNode &cl = F[S[i]];
        cl.virtualAddLiter();
    }
}

}  // namespace SAT
