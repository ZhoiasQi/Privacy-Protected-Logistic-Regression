#include "logistic_regression.hpp"

using namespace Eigen;
using Eigen::Matrix;
using namespace emp;
using namespace std;

void LogisticRegression::train_model(){
    for(int i = 0; i < t; i++){
        int indexLo = (i * BATCH_SIZE) % n;  
        online->train_batch(i, indexLo);
    }

    if (party == BOB){
        send<ColVectorXi64>(io, online->wi);
    }
    else{
        recv<ColVectorXi64>(io, w);
    }
        
    if (party == ALICE){
        send<ColVectorXi64>(io, online->wi);
    }
    else{
        recv<ColVectorXi64>(io, w);
    }

    w += online->wi;

    descale<ColVectorXi64, ColVectorXd>(w, w_d);
}