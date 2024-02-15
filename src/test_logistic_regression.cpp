#include "test_logistic_regression.hpp"

void TestLogisticRegression::getW(ColVectorXi64 w){
    this->w = w;
}

void TestLogisticRegression::secret_share_w(){
    
    if (party == ALICE) {  // �����ǰ���뷽��ALICE
        emp::PRG prg;  // α���������������
        ColVectorXi64 rW(w.rows(), w.cols());  // �����ǩ����
        prg.random_data(rW.data(), w.rows() * w.cols() * sizeof(uint64_t));  // ����������� rW
        wi = w + rW;
        rW *= -1;
        send<ColVectorXi64>(io, rW);  // ����������� rY

        cout << "Alice has secretly sent the model w to Carol" << endl;
    } 
    else {  // �����ǰ���뷽��Alice
        recv<ColVectorXi64>(io, wi);  // �ӶԷ����ռ��ܺ��ѵ����ǩ

        cout << "Carol has received the secret data from Alice" << endl;

    }
}

void TestLogisticRegression::test_model(){

    for(int i = 0; i < t; i++){
        int indexLo = (i * n) % n;  
        online->test_model(i, indexLo);
    }

    if (party == ALICE){
        send<ColVectorXi64>(io, online->prediction_i);
        
        cout << "Alice has already sent prediction_i to Carol" << endl;
    }
    else{
        recv<ColVectorXi64>(io, w);

        cout << "Carol has already got prediction_i from Alice" << endl;
    }

    if(party == CAROL){
        prediction += online->prediction_i;

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