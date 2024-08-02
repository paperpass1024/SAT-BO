#include "solver.h"
#include <iostream>
#include <algorithm>
using namespace std;

namespace SAT 
{
    int Solver::getlevel(sol &ans,int num,int allnum){
            int prelevel=ans.decisionLevl.size();
            vector<int> nullSearch;
            int j=0;
            vector<int> level;
            for(auto x:ans.decisionLevl){
                ++j;
                if(ans.isSearch[x]) continue;
                nullSearch.push_back(x);
                level.push_back(j);
            }
            if(2*num>allnum&&nullSearch.size()>200){
                int p=rand(100)+1;
                if(p<10*num/allnum){
                    prelevel=level[rand(level.size())];
                }
            }
            return prelevel;
    }
    bool Solver::dpll(sol &ans,vector<int> q,vector<VariableValue>& output,std::function<bool()> isTimeout){
        vector<int> tmp;
        if(isTimeout()){return false;}
        if(!ans.unit_propagation(q,tmp)){return false;}
        if(ans.isSatisfied()){
            bool flag=false;
            randomForIndependenceVar(ans);
            ans.getAns(output[outputs]);
            string s;
            VariableValue & t=output[outputs];
            for (int i=1;i<t.size();i++){
                s=s+(char)((int)'0'+t[i]);
            }
            if(mp.find(s)==mp.end()) {
                mp[s]=1;
                outputs++;
                limit_l++;
                if(limit_l==limit){ flag=true;}
            }
            ans.recovery(q,tmp);
            return flag;
        }
        vector<VariableId> can;
        int t=ans.getPreferAssignment(can);
        for(auto &i :can) {
            int id=abs(i);
            Node &node=ans.X[id];
            if(rand(100)>node.assignmentPrefer*10){
                i=-id;
            }
            else i=id;
        }
            /* if(rand(100)<50) i=0-i; */
        /* if(t==0){ */
            /* for(auto &i :can) if(rand(100)<50) i=0-i; */
        /* } */
        /* ans.getCandicate(can); */
        VariableId x=can[rand(can.size())];
        bool flag=(dpll(ans,{x},output,isTimeout)||dpll(ans,{-x},output,isTimeout));
        ans.recovery(q,tmp);
        return flag;
    }
    void Solver::randomForIndependenceVar(sol &ans){
        for(int i=0;i<ans.X_len;i++){
            int id=ans.X_id[i];
            Node &node=ans.X[id];
            if(rand(100)>node.assignmentPrefer*10){
                ans.ans[id]=0;
            }
            else ans.ans[id]=1;
            /* if(rand(100)<10) */
            /*     ans.ans[id]=rand(100)%2; */
            /* else{ */
            /*     if(node.assignmentPrefer==5) ans.ans[id]=rand(2); */
            /*     else ans.ans[id]=(node.assignmentPrefer>5?1:0); */
            /* } */
        }
    }
    bool Solver::isInvert(int x,sol &ans){
        double perheps=(5.0/max(0.5,1.0*abs(ans.getNode(x).assignmentPrefer-5)));
        int pp=rand(100)+1;
        return (pp<perheps);
    }
    bool Solver::dpllForAssignment(sol &ans,vector<int> q,vector<VariableValue>& output,std::function<bool()> isTimeout){
        vector<int> tmp;
        if(limit_l==limit) return true;
        if(isTimeout()){return false;}
        if(!ans.unit_propagation(q,tmp)){
            preLevel=getlevel(ans,limit_l,limit);
            return false;
        }
        if(ans.isSatisfied()){
            bool flag=false;
            randomForIndependenceVar(ans);
            ans.getAns(output[outputs]);
            string s;
            VariableValue & t=output[outputs];
            for (int i=1;i<t.size();i++){
                s=s+(char)((int)'0'+t[i]);
            }
            if(mp.find(s)==mp.end()) {
                mp[s]=1;
                outputs++;
                limit_l++;
                if(limit_l==limit){ flag=true;}
            }
            ans.recovery(q,tmp);
            return flag;
        }
        // get candicate
        vector<VariableId> can;
        /* int t=ans.getPreferAssignment(can); */
        /* if(t==0){ */
        /*     for(auto &i :can) if(rand(100)<50) i=0-i; */
        /* } */
        int t=ans.getPreferAssignment(can);
        for(auto &i :can) {
            int id=abs(i);
            Node &node=ans.X[id];
            if(rand(100)>node.assignmentPrefer*10){
                i=-id;
            }
            else i=id;
        }

        VariableId x=can[rand(can.size())];
        if(isInvert(x,ans)) x=-x;

        // continue to search
        ans.decisionLevl.push_back(abs(x));
        if(dpllForAssignment(ans,{x},output,isTimeout)) {
            ans.recovery(q,tmp);
            return true;
        }
        if(preLevel&&preLevel!=ans.decisionLevl.size()) {
            // cerr<<"ok"<<endl;
            // cerr<<preLevel<<" "<<ans.decisionLevl.size()<<endl;
            ans.decisionLevl.pop_back();
            ans.recovery(q,tmp);
            return false;
        }
        if(preLevel==ans.decisionLevl.size()){
            // cerr<<"okk"<<endl;
        }
        ans.isSearch[abs(x)]=true;
        preLevel=0;
        if(dpllForAssignment(ans,{-x},output,isTimeout)) {
            ans.recovery(q,tmp);
            return true;
        }
        if(preLevel==ans.decisionLevl.size()){
            preLevel=0;
        }
        ans.decisionLevl.pop_back();
        ans.isSearch[abs(x)]=false;
        ans.recovery(q,tmp);
        if(!preLevel) {
            preLevel=getlevel(ans,limit_l,limit);
        }
        // cerr<<"ok:"<<(preLevel)<<" "<<ans.decisionLevl.size()<<endl;;
        return false;
    }
    bool Solver::solve(vector<VariableValue>& output, SATProblem& input, std::function<bool()> isTimeout, int seed,bool isHasAssignmentPrefer)
    {
        initRand(seed);
        limit_l=outputs=0;
        if(isHasAssignmentPrefer){ 
            int i=0;
            /* sol &ans=ansers[i]; */
            /* if(!ans.init(input,input.assignmentPrefers[0])) {return false;} */
            /* for (i=0;i<input.solUpLimit&&outputs<input.solUpLimit;i++){ */
                /* limit=2; */
                /* limit_l=0; */
                /* dpllForAssignment(ans,{},output,isTimeout); */
            /* } */

            for(auto assignmentPrefer:input.assignmentPrefers){
                limit=input.perUpLimit[i];
                limit_l=0;
                /* outputs=0; */
                sol &ans=ansers[i];
                if(!ans.init(input,assignmentPrefer)) {return false ;}
                dpllForAssignment(ans,{},output,isTimeout);
                cerr<<"solve one assignmentPrefer "<<i<<"\n";
                i++;

            }
        }
        else {

            limit=input.solUpLimit;
            sol &ans=ansers[0];
            if(!ans.init(input)) {return false;}
            for (int i=0;i<input.solUpLimit&&outputs<input.solUpLimit;i++){
                limit=1;
                limit_l=0;
                dpll(ans,{},output,isTimeout);
            }
        }
        while(output.size()>outputs) output.pop_back();

        return check(output,input);
    }
	

	// solver.
	bool solveSATProblem(vector<VariableValue>& output, SATProblem& input, std::function<bool()> isTimeout, int seed,bool isHasAssignmentPrefer)
	{
		return Solver().solve(output, input, isTimeout, seed,isHasAssignmentPrefer);
		// cerr << "end\n";
	}

}
