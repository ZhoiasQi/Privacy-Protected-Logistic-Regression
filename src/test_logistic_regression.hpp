#ifndef T_LOGISTIC_REGRESSION_H
#define T_LOGISTIC_REGRESSION_H

#include "test_offline_phase.hpp"
#include "test_online_phase.hpp"
#include <iostream>

using namespace std;

class TestLogisticRegression{
public:
    emp::NetIO* io; 
    int party;  
    int n, d, t;  
    RowMatrixXi64 X; 
    RowMatrixXi64 Xi;
    ColVectorXi64 Y;  
    ColVectorXi64 w; 
    ColVectorXi64 wi;
    ColVectorXi64 prediction;
    ColVectorXi64 prediction_i;
    ColVectorXd predictionD;
    TestingParams params;
    SetupTriples triples;
    TestSetUp* setup;
    TestOnlinePhase* online; 

    TestLogisticRegression(RowMatrixXi64& testing_data, ColVectorXi64& testing_labels, TestingParams params, emp::NetIO* io) {
        this->n = params.n;
        this->d = params.d;
        this->t = 1;
        this->X = testing_data;
        this->Y = testing_labels;
        this->io = io;
        this->party = PARTY;
        this->w.resize(d);
        this->wi.resize(d);
        this->prediction.resize(n);
        this->prediction_i.resize(n);
        this->predictionD.resize(n);
        this->params = params;
        
        this->setup = new TestSetUp(n, d, t, io);
        setup->generateMTs();

        SetupTriples triples;
        setup->getMTs(&triples);
        this->triples = triples;

        RowMatrixXi64 Xi(X.rows(), X.cols());  
        //ColVectorXi64 Yi(Y.rows(), Y.cols()); 

        if (party == CAROL) {  
            emp::PRG prg;  
            RowMatrixXi64 rX(X.rows(), X.cols()); 
            //ColVectorXi64 rY(Y.rows(), Y.cols()); 
            prg.random_data(rX.data(), X.rows() * X.cols() * sizeof(uint64_t)); 
            //prg.random_data(rY.data(), Y.rows() * Y.cols() * sizeof(uint64_t)); 
            Xi = X + rX;  
            //Yi = Y + rY;  
            rX *= -1;  
            //rY *= -1;  
            send<RowMatrixXi64>(io, rX); 
            //send<ColVectorXi64>(io, rY);  

            cout << "Carol has secretly sent the data to Alice" << endl;
        } else {  
            recv<RowMatrixXi64>(io, Xi);  
            //recv<ColVectorXi64>(io, Yi); 

            cout << "Alice has received the secret data from Carol" << endl;

        }

        this->Xi = Xi;
    }

    void getW(ColVectorXi64 w);

    void secret_share_w();

    double test_model();

    double testmodel();
};

#endif

