#ifndef OFFLINE_PHASE_H
#define OFFLINE_PHASE_H

#include <emp-ot/emp-ot.h>
#include "util.hpp"

using namespace emp;
using namespace std;

class OfflineSetUp{
public:
    int n, d, t;
    int party;
    RowMatrixXi64 Ai;  
    ColMatrixXi64 Bi;  
    ColMatrixXi64 Ci;  
    ColMatrixXi64 Bi_;  
    ColMatrixXi64 Ci_; 

    NetIO* io; 
    SHOTExtension<emp::NetIO>* send_ot; 
    SHOTExtension<emp::NetIO>* recv_ot; 
    PRG prg; 

    OfflineSetUp(int n, int d, int t, NetIO* io){
        this->n = n;
        this->d = d;
        this->t = t;
        this->party = PARTY;
        this->io = io;
        this->send_ot = new SHOTExtension<NetIO>(io);
        this->recv_ot = new SHOTExtension<NetIO>(io);

        Ai.resize(n, d); 
        Bi.resize(d, t); 
        Ci.resize(BATCH_SIZE, t);  
        Bi_.resize(BATCH_SIZE, t);  
        Ci_.resize(d, t);  

        initialize_matrices();  
        cout << "Matrices Initialized" << endl;  
    }

    void initialize_matrices();
    void generateMTs();
    void secure_mult(int N, int D, vector<vector<uint64_t>>& a, vector<uint64_t>& b, vector<uint64_t> &c);
    void getMTs(SetupTriples *triples);
    void verify();
};

#endif