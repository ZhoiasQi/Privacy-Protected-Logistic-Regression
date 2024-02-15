#include "test_logistic_regression.hpp"

void TestLogisticRegression::getW(ColVectorXi64 w){
    this->w = w;
}

void TestLogisticRegression::secret_share_w(){
    
    if (party == ALICE) {  
        emp::PRG prg;  // 伪随机数生成器对象
        ColVectorXi64 rW(w.rows(), w.cols());  // 随机标签向量
        prg.random_data(rW.data(), w.rows() * w.cols() * sizeof(uint64_t));  // 生成随机数据 rW
        wi = w + rW;
        rW *= -1;
        send<ColVectorXi64>(io, rW);  // 发送随机数据 rY

        cout << "Alice has secretly sent the secret model w to Carol" << endl;
    } 
    else {  

        recv<ColVectorXi64>(io, wi);  // 从对方接收加密后的训练标签

        cout << "Carol has received the secret model w from Alice" << endl;

    }
}

void TestLogisticRegression::test_model(){
    
    this->online = new TestOnlinePhase(params, io, &triples);

    this->online->initialize(this->Xi, this->wi);

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
    }

    if(party == CAROL){

        //cout << online->prediction_i.size() << endl;

        //cout << prediction.size() << endl;

        prediction = online->prediction_i + prediction;

        int n_ = prediction.size();

        int num_correct = 0;

        for (int i = 0; i < n_; i++){
            if(Y[i] == SCALING_FACTOR){
                if(prediction[i] >= (SCALING_FACTOR >> 1)){
                    num_correct++;
                }
            }
            else{
                if(prediction[i] < (SCALING_FACTOR >> 1)){
                    num_correct++;
                }
            }
        }

    double accuracy = num_correct/((double) n_);
    cout << "Accuracy on testing the trained model is " << accuracy * 100 << endl;

    }
}