#include "test_logistic_regression.hpp"

using namespace std;


void TestLogisticRegression::getW(ColVectorXi64 w){
    this->w = w;
}

void TestLogisticRegression::secret_share_w(){
    
    if (party == ALICE) {  
        emp::PRG prg;  
        ColVectorXi64 rW(w.rows(), w.cols()); 
        prg.random_data(rW.data(), w.rows() * w.cols() * sizeof(uint64_t));  
        wi = w + rW;
        rW *= -1;
        send<ColVectorXi64>(io, wi);  
        wi = rW;

        cout << "Alice has secretly sent the secret model w to Carol" << endl;
    } 
    else {  

        recv<ColVectorXi64>(io, wi);  

        cout << "Carol has received the secret model w from Alice" << endl;

    }
}

double TestLogisticRegression::test_model(){
    
    this->online = new TestOnlinePhase(params, io, &triples);

    this->online->initialize(this->Xi, this->wi);

    //auto x = ran();
    
    for(int i = 0; i < t; i++){
        int indexLo = (i * n) % n;  
        online->test_model(i, indexLo);
    }

    if (party == ALICE){
        send<ColVectorXi64>(io, online->prediction_i);
        
        cout << "Alice has already sent prediction_i to Carol" << endl;
        //cout << online->prediction_i.size() << endl;
    }
    else{
        recv<ColVectorXi64>(io, prediction);

        cout << "Carol has already got prediction_i from Alice" << endl;
        //cout << prediction.size() << endl;

        prediction = online->prediction_i + prediction;

        int n_ = prediction.size();

        descaleTest<ColVectorXi64, ColVectorXd>(prediction, predictionD);

        for(int i = 0; i < n_; i++){
            double temp;
            // cout << prediction[i] << endl;
            //cout << predictionD[i] << endl;
            if(predictionD[i] >= 1){
                temp = 1.0;
            }
            else if(predictionD[i] >= -1 && predictionD[i] < 1){
                temp = predictionD[i] + 0.5;
            }
            else{
                temp = 0;
            }
            //temp = 1.0 / (1 + exp(-predictionD[i]));
            predictionD[i] = temp;
        }
        
        int num_correct = 0;

        for (int i = 0; i < n_; i++){
            if(Y[i] == SCALING_FACTOR){
                if(predictionD[i] >= 0.5){
                    num_correct++;
                    cout << 1 << endl;
                }
            }
            else{
                if(predictionD[i] < 0.5){
                    num_correct++;
                    cout << 0 << endl;
                }
            }
        }

        double accuracy = num_correct/((double) n_);

        return accuracy;

    }

}