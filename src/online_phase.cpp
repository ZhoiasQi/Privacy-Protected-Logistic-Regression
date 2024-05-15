#include "online_phase.hpp"

using namespace Eigen;
using Eigen::Matrix;
using namespace emp;
using namespace std;

extern Traffic traffic;

void OnlinePhase::initialize(RowMatrixXi64& Xi, ColVectorXi64& Yi){
    this->Xi = Xi;
    this->Yi = Yi;

    for(int i = 0; i < d; i++){
        wi(i) = 0;
    }

    Ui = triples->Ai;
    Ei = Xi - Ui;
    Vi = triples->Bi;
    Vi_ = triples->Bi_;
    Zi = triples->Ci;
    Zi_ = triples->Ci_;

    if(party == ALICE){
        send<RowMatrixXi64>(io, Ei);
    }
    else{
        recv<RowMatrixXi64>(io, E);
    }

    if(party == BOB){
        send<RowMatrixXi64>(io, Ei);
    }
    else{
        recv<RowMatrixXi64>(io, E);
    }

    traffic.online += (sizeof(Ei) + sizeof(E))  / (double)B_TO_MB; 

    E = E + Ei;
}

void OnlinePhase::train_batch(int iter, int indexLo){
    
    RowMatrixXi64 X = Xi.block(indexLo, 0, BATCH_SIZE, d); 
    ColVectorXi64 Y = Yi.segment(indexLo, BATCH_SIZE); 
    RowMatrixXi64 Eb = E.block(indexLo, 0, BATCH_SIZE, d); 
    ColVectorXi64 V = Vi.col(iter); 
    ColVectorXi64 V_ = Vi_.col(iter); 
    ColVectorXi64 Z = Zi.col(iter);
    ColVectorXi64 Z_ = Zi_.col(iter);

    Fi = wi - V;

    ColVectorXi64 D(BATCH_SIZE);
    ColVectorXi64 Y_(BATCH_SIZE);
    ColVectorXi64 Sig(BATCH_SIZE);
    ColVectorXi64 Fi_(BATCH_SIZE);
    ColVectorXi64 F_(BATCH_SIZE);
    ColVectorXi64 delta(d);

    if(party == ALICE){
        send<ColVectorXi64>(io, Fi);
    }
    else{
        recv<ColVectorXi64>(io, F);
    }

    if(party == BOB){
        send<ColVectorXi64>(io, Fi);
    }
    else{
        recv<ColVectorXi64>(io, F);
    }

    traffic.online += (sizeof(Fi) + sizeof(F))  / (double)B_TO_MB; 

    F = F + Fi;

    Y_ = -i * (Eb * F) + X * F + Eb * wi + Z;

    truncate<ColVectorXi64>(i, SCALING_FACTOR, Y_);

    Sig = sigmoid(Y_, i, io);

    D = Sig - Y;

    Fi_ = D - V_;

    if (party == ALICE){
        send<ColVectorXi64>(io, Fi_);
    }
    else{
        recv<ColVectorXi64>(io, F_);
    }

    if (party == BOB){
        send<ColVectorXi64>(io, Fi_);
    }
    else{
        recv<ColVectorXi64>(io, F_);
    }

    traffic.online += (sizeof(Fi_) + sizeof(F_))  / (double)B_TO_MB; 

    F_ = F_ + Fi_;

    RowMatrixXi64 Xt = X.transpose();
    RowMatrixXi64 Ebt = Eb.transpose();

    delta = -i * (Ebt * F_) + Xt * F_ + Ebt * D + Z_;

    truncate<ColVectorXi64>(i, SCALING_FACTOR, delta);

    // int min = std::min(iter, 15);
    //cout << min;
    // auto k = alpha_inv;

    // if(iter < 5){
    //     k = alpha_inv;
    // }
    // else if(iter < 45){
    //     int e = (iter - 5) / 5;
    //     k = Mypow(k, e);
    // }
    // else{
    //     k = Mypow(k, 8);
    // }

    // if(iter / 5 < 10){
    //     k = Mypow(k, (iter / 5) + 1);
    // }
    // else{
    //     k = Mypow(k, 6);
    // }
    
    //cout << k << " ";

    // truncate<ColVectorXi64>(i, k * BATCH_SIZE, delta);
    truncate<ColVectorXi64>(i, alpha_inv * BATCH_SIZE, delta);

    wi = wi - delta;

}

uint64_t Mypow(uint64_t a, int b){
    for(int i = 0; i < b; i++){
        a *= 2;
    }
    return a;
}