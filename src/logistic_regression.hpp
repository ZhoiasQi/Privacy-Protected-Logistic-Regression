#ifndef LOGISTIC_REGRESSION_H
#define LOGISTIC_REGRESSION_H

#include "offline_phase.hpp"
#include "online_phase.hpp"

class LogisticRegression{
public:
    int party;

    LogisticRegression(){
        this->party = PARTY;
    }

    void train_model();

};

#endif