#include "linear_regression.hpp"

using namespace Eigen;
using Eigen::Matrix;
using namespace emp;
using namespace std;

void LinearRegression::train_model(){
    for (int i = 0; i < t; i++){
        int indexLo = (i * BATCH_SIZE) % n;  // �����������½�
        online->train_batch(i, indexLo);  // ����ѵ����������
    }

    if (party == BOB){
        send<ColVectorXi64>(io, online->wi);  // ����� BOB ������������ѧϰ��Ȩ������ wi
    }
    else
        recv<ColVectorXi64>(io, w);  // ����� ALICE ������������ѧϰ��Ȩ������ w

    if (party == ALICE){
        send<ColVectorXi64>(io, online->wi);  // ����� ALICE ������������ѧϰ��Ȩ������ wi
    }
    else
        recv<ColVectorXi64>(io, w);  // ����� BOB ������������ѧϰ��Ȩ������ w

    w += online->wi;  // ����Ȩ������ w

    descale<ColVectorXi64, ColVectorXd>(w, w_d);  // �� w ���з����Ų������õ� w_d
}

void LinearRegression::test_model(RowMatrixXd& testing_data, ColVectorXd& testing_labels){
    ColVectorXd prediction;  // �洢Ԥ����
    prediction = testing_data * w_d;  // ʹ��ѵ���õ���Ȩ�� w_d ����Ԥ��
    prediction *= 10;  // ��Ԥ�����������Ŵ������� 10
    int n_ = testing_labels.rows();  // ��ȡ�������ݵ�������

    ColVectorXd error;  // �洢���
    prediction = round(prediction.array());  // ��Ԥ����������������

    int num_correct = 0;  // ��ȷ�����������
    for (int i = 0; i < n_; i++){
        if(prediction[i] == testing_labels[i])  // ���Ԥ�����Ƿ�����Ա�ǩһ��
            num_correct++;  // ��һ�£�����ȷ��������һ
    }
    double accuracy = num_correct/((double) n_);  // ����׼ȷ��
    cout << "Accuracy on testing the trained model is " << accuracy * 100 << endl;  // ������Լ��ϵ�׼ȷ��
}
