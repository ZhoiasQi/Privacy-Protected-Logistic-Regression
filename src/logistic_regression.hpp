#ifndef LOGISTIC_REGRESSION_H
#define LOGISTIC_REGRESSION_H

#include "offline_phase.hpp"
#include "online_phase.hpp"
#include <iostream>

using namespace std;

class LogisticRegression{
public:
    emp::NetIO* io;  
    int party; 
    int n, d, t; 
    RowMatrixXi64 X;  
    ColVectorXi64 Y;  
    ColVectorXi64 w; 
    ColVectorXd w_d;  
    OfflineSetUp* setup; 
    OnlinePhase* online;  

    //ѵ���׶ι��캯��
    LogisticRegression(RowMatrixXi64& training_data, ColVectorXi64& training_labels, TrainingParams params, emp::NetIO* io) {
        this->n = params.n;  
        this->d = params.d; 
        this->t = (params.n) / BATCH_SIZE; 
        this->X = training_data; 
        this->Y = training_labels;  
        this->io = io; 
        this->party = PARTY;  
        this->w.resize(d);  
        this->w_d.resize(d);  

        this->setup = new OfflineSetUp(n, d, t, io);
        setup->generateMTs(); 

        SetupTriples triples;
        setup->getMTs(&triples);  

        RowMatrixXi64 Xi(X.rows(), X.cols()); 
        ColVectorXi64 Yi(Y.rows(), Y.cols());  
        
        if (party == emp::ALICE) {  
            emp::PRG prg;  
            RowMatrixXi64 rX(X.rows(), X.cols());  
            ColVectorXi64 rY(Y.rows(), Y.cols());  
            prg.random_data(rX.data(), X.rows() * X.cols() * sizeof(uint64_t));  
            prg.random_data(rY.data(), Y.rows() * Y.cols() * sizeof(uint64_t));  
            Xi = X + rX;  
            Yi = Y + rY;  
            rX *= -1;  
            rY *= -1;  
            send<RowMatrixXi64>(io, rX); 
            send<ColVectorXi64>(io, rY);  

            cout << "Alice has secretly sent the data to Bob" << endl;

        } else { 
            recv<RowMatrixXi64>(io, Xi); 
            recv<ColVectorXi64>(io, Yi);  

            cout << "Bob has received the secret data from Alice" << endl;
        }

        this->online = new OnlinePhase(params, io, &triples);  
        online->initialize(Xi, Yi); 

        train_model();

    }

    void train_model();

    void test_model(RowMatrixXd& testing_data, ColVectorXd& testing_labels);
};

#endif