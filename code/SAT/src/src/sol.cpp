#include "sol.h"
#include <algorithm>
#include <cstdlib>
// =====================================================================================
namespace SAT{
bool sol::unit_propagation(vector<LiteralId> &decisionLevel, vector<ClauseId> &changeClauseSet) {
    if (!single_unit_propagation(decisionLevel, changeClauseSet))
        return false;
    for (auto i : decisionLevel) {
        vitrualRemoveNode(i);
        ans[getIterId(i)] = (i > 0);
    }
    for (auto clid : changeClauseSet) {
        int originLen = abs(vis[clid]);
        remove(originLen, clid);
        ClauseNode &cl = F[clid];
        if (vis[clid] < 0) continue;
        insert(cl.len, clid);
        vis[clid] = 0;
    }
    return true;
}
void sol::recovery(vector<LiteralId> &decisionLevel, vector<ClauseId> &changeClauseSet) {
    for (long unsigned int i = 0; i < decisionLevel.size(); i++)
        vitrualAddNode();
    for (auto i : changeClauseSet)
        if (vis[i] == 0) remove(F[i].len, i);
    singleRecovery(decisionLevel, changeClauseSet);
    for (auto i : changeClauseSet) insert(F[i].len, i);
}
bool sol::single_unit_propagation(vector<LiteralId> &decisionLevel, vector<ClauseId> &changeClauseSet) {
    for (auto i : decisionLevel) pos[getIterId(i)] = true;
    for (long unsigned int i = 0; i < decisionLevel.size(); i++) {
        LiteralId liter = decisionLevel[i];
        Node &node = getNode(liter);
        removeSatisfiedClauses(node.S[liter > 0], liter, decisionLevel, changeClauseSet);
        removeUnsatisfiedClauses(node.S[liter < 0], liter, decisionLevel, changeClauseSet);
        if (isHasEmptyClause) {
            isHasEmptyClause = false;
            while (decisionLevel.back() != liter) {
            pos[getIterId(decisionLevel.back())] = false;
            decisionLevel.pop_back();
            }
            singleRecovery(decisionLevel, changeClauseSet);
            return false;
        }
    }
    return true;
}
void sol::singleRecovery(vector<LiteralId> &decisionLevel, vector<ClauseId> &changeClauseSet) {
    reverse(decisionLevel.begin(), decisionLevel.end());
    for (auto liter : decisionLevel) {
        Node &node = getNode(liter);
        pos[getIterId(liter)] = false;
        addUnsatisfiedClauses(node.S[liter < 0]);
        addSatisfiedClauses(node.S[liter > 0], liter);
    }
    for (auto i : changeClauseSet) vis[i] = 0;
}
int sol::getPreferAssignment(vector<VariableId> &can) {
    int mx = 0;
    for (int i = 0; i < X_len; i++) {
        Node &node = getNode(X_id[i]);
        int tmp = abs(node.assignmentPrefer - 5);
        if(tmp<mx) continue ;
        if (tmp > mx) {
            mx = tmp;
            can.clear();
        }
        if (node.assignmentPrefer > 5)
            can.push_back(X_id[i]);
        else can.push_back(-X_id[i]);
    }
    return mx;
}
void sol::getCandicate(vector<VariableId> &can) {
    if (F_len == 0) {
        can.push_back(X_id[0]);
        return;
    }
    for (MinLen = 2; MinLen <= n; MinLen++)
        if (bucketOfFLen[MinLen].size()) break;

    vector<int> tmp(X_len, 0);
    int mx = 0;
    for (int len = MinLen; len <= max(MinLen, min(3, (int)(bucketOfFLen.size() - 1))); len++) {
        int bas = (len == 2 ? 5 : 1);
        for (auto clid : bucketOfFLen[len]) {
            ClauseNode &cl = F[clid];
            for (int j = 1; j <= len; j++) {
                LiteralId i = getNode(cl.C[j]).name;
                tmp[i] += bas;
            }
        }
    }
    for (int i = 0; i < X_len; i++) mx = max(mx, tmp[i]);
    for (int i = 0; i < X_len; i++)
    if (mx == tmp[i]) can.push_back(X_id[i]);
}
}
