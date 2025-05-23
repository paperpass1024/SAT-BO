#include "SAT.h"
namespace SAT{
class Node {
public:
    VariableId name;
    Clause S[2], id;
    int assignmentPrefer;
    inline bool isEmpty();
    inline void init(ClauseId cln, VariableId nameInit);
    inline void addS(ClauseId clid, bool val);
    inline void removeS(ClauseId clid, bool val);
    inline void virtualRemoveS(ClauseId clid, LiteralId liter);
    inline void virtualAddS(LiteralId liter);
};
class ClauseNode {
public:
    Clause C, id;
    VariableId len = 0;
    inline bool createClause(Clause clause, VariableId n);
    inline bool isUnitClause();
    inline bool isEmpty();
    inline void removeLiteral(LiteralId liter);
    inline int getIterId(LiteralId x);
    inline void virtualRemoveLiter(LiteralId x);
    inline void virtualAddLiter();
};
inline bool Node::isEmpty() { return ((!S[0][0]) && (!S[1][0])); }
inline void Node::init(ClauseId cln, VariableId nameInit) {
    name = nameInit;
    S[0].push_back(0);
    S[1].push_back(0);
    id.resize(cln);
}
inline void Node::addS(ClauseId clid, bool val) {
    S[val].push_back(clid);
    ++S[val][0];
    id[clid] = S[val][0];
}
inline void Node::removeS(ClauseId clid, bool val) {
    S[val][id[clid]] = S[val].back();
    id[S[val].back()] = id[clid];
    id[clid] = 0;
    --S[val][0];
    S[val].pop_back();
}
inline void Node::virtualRemoveS(ClauseId clid, LiteralId liter) {
    bool val = (liter > 0);
    VariableId en = S[val][S[val][0]];
    std::swap(id[clid], id[en]);
    std::swap(S[val][id[clid]], S[val][id[en]]);
    S[val][0]--;
}
inline void Node::virtualAddS(LiteralId liter) { S[liter > 0][0]++; }

inline bool ClauseNode::createClause(Clause clause, VariableId n) {
    C.clear();
    C.push_back(0);
    len = 0;
    id.clear();
    id.resize(n + 1, 0);
    for (auto iter : clause) {
        int idName = getIterId(iter);
        if (id[idName]) {
            if (C[id[idName]] == iter)
                continue;
            else {
                return false;
            }
        }
        len++;
        id[idName] = len;
        C.push_back(iter);
    }
    return true;
}
inline bool ClauseNode::isUnitClause() { return (len == 1); }
inline bool ClauseNode::isEmpty() { return (len == 0); }
inline void ClauseNode::removeLiteral(LiteralId liter) {
    LiteralId en = getIterId(C[len]);
    liter = getIterId(liter);
    C[id[liter]] = C[len];
    id[en] = id[liter];
    len--;
    C.pop_back();
}
inline int ClauseNode::getIterId(LiteralId x) {
    return (x ^ (x >> 31)) - (x >> 31);
}
inline void ClauseNode::virtualRemoveLiter(LiteralId x) {
    VariableId en = getIterId(C[len]);
    x = getIterId(x);
    std::swap(id[x], id[en]);
    std::swap(C[id[x]], C[id[en]]);
    len--;
}
inline void ClauseNode::virtualAddLiter() { len++; }

}
