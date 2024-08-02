#include "../../include/SAT.h"
#include <stdlib.h>

#include <chrono>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <iostream>
#include <istream>
#include <sstream>

using namespace std;
using namespace SAT;

void loadInput(istream &is, SATProblem &gc) {
    cerr << ">>>>>load input." << endl;
    char type, tmp[10000];
    is >> type;
    while (type == 'c') {
        is.getline(tmp, 10000);
        is >> type;
    }
    is >> tmp;
    is >> gc.num >> gc.clauseNum;
    gc.C.resize(gc.clauseNum);
    for (auto &cln : gc.C) {
        int t;
        is >> t;
        while (t != 0) {
            cln.push_back(t);
            is >> t;
        }
    }
    cerr<<"<<<<<load over.\n";
}
void saveOutput(ostream &os, vector<VariableValue> &Xs, int varnum) {
    cerr << ">>>>>save solves:\n";
    os << "s" << " " << varnum << " " << Xs.size() << endl;
    for (auto X : Xs) {
        for (int i = 1; i <= varnum; i++) {
            if (X[i]) os << i << " ";
            else os << -i << " ";
        }
        os << "0" << endl;
    }
    cerr << "solves has save in file!" << endl;

}
void loadOneAssignmentPrefer(istringstream &is,Prefers &p){
    double t;
    while(is>>t){
        p.push_back(t);
    }
}
void loadAssignmentPrefer(istream &is, SATProblem &gc ,int solUpper) {
    cerr << ">>>>>load assignment prefers." << endl;
    int num=0;
    string s;
    getline(is,s);
    num=atoi(&s[0]);
    gc.assignmentPrefers.resize(num);
    for (int i=0; i<num;i++){
        getline(is,s);
        istringstream iss(s);
        loadOneAssignmentPrefer(iss,gc.assignmentPrefers[i]);
    }
    // every prefer solve UpLimit
    if (gc.assignmentPrefers.size() == 2)
        gc.perUpLimit = {solUpper / 2, solUpper - solUpper / 2};
    else
        gc.perUpLimit = {solUpper};

}
void getInfo(clock_t runTime,bool isTimeOut,bool isSatisfied,int solNum){
    cerr << ">>>>>save output.\n";
    cerr << "time = " << double(runTime) / CLOCKS_PER_SEC << " seconds\n";
    if (isTimeOut) cerr << "ans = timeout" << endl;
    else cerr << "ans = " << isSatisfied << endl;
    cerr << "solNum = "<<solNum<<endl;
    //// solves

}
void test(istream &is, ostream &os, istream &vm, long long secTimeout, int randSeed, bool isHasAssignment, int solUpLimit) {
    SATProblem SAT;
    loadInput(is, SAT);
    if (isHasAssignment) {
        loadAssignmentPrefer(vm, SAT, solUpLimit);
    }
    SAT.solUpLimit = solUpLimit;
    chrono::steady_clock::time_point endTime = chrono::steady_clock::now() + chrono::seconds(secTimeout);
    vector<VariableValue> variableValues(solUpLimit,VariableValue(SAT.num+1));
    clock_t start = clock();
    bool isSatisfied=solveSATProblem(variableValues, SAT, [&]() -> bool { return endTime < chrono::steady_clock::now(); }, randSeed, isHasAssignment);
    clock_t end = clock();

    // save info of sat 
    getInfo(end-start, endTime < chrono::steady_clock::now(),isSatisfied,variableValues.size());
    //solves
    saveOutput(os, variableValues, SAT.num);
}

int main(int argc, char *argv[]) {
    srand(time(0));
    if(argc!=5&&argc!=6) return 0;
    ifstream ifs(argv[1], ios::in);
    ofstream ofs(argv[2], ios::out);
    ifstream vm;
    long long secTimeout = 0;
    int solNum = 0;
    if (argc == 5) {
        secTimeout = atoll(argv[3]);
        solNum = atoi(argv[4]);
    } else if (argc == 6) {
        vm=ifstream(argv[3], ios::in);
        secTimeout = atoll(argv[4]);
        solNum = atoi(argv[5]);
    }
    test(ifs, ofs, vm, secTimeout, static_cast<int>(time(nullptr) + clock()),argc==6,solNum);
    return 0;
}
