#include "linear_regression.hpp"

using namespace Eigen;
using Eigen::Matrix;
using namespace emp;
using namespace std;

void LinearRegression::train_model(){
    for (int i = 0; i < t; i++){
        int indexLo = (i * BATCH_SIZE) % n;  // 计算索引的下界
        online->train_batch(i, indexLo);  // 在线训练批次数据
    }

    if (party == BOB){
        send<ColVectorXi64>(io, online->wi);  // 如果是 BOB 方，发送在线学习的权重向量 wi
    }
    else
        recv<ColVectorXi64>(io, w);  // 如果是 ALICE 方，接收在线学习的权重向量 w

    if (party == ALICE){
        send<ColVectorXi64>(io, online->wi);  // 如果是 ALICE 方，发送在线学习的权重向量 wi
    }
    else
        recv<ColVectorXi64>(io, w);  // 如果是 BOB 方，接收在线学习的权重向量 w

    w += online->wi;  // 更新权重向量 w

    descale<ColVectorXi64, ColVectorXd>(w, w_d);  // 对 w 进行反缩放操作，得到 w_d
}

void LinearRegression::test_model(RowMatrixXd& testing_data, ColVectorXd& testing_labels){
    ColVectorXd prediction;  // 存储预测结果
    prediction = testing_data * w_d;  // 使用训练得到的权重 w_d 进行预测
    prediction *= 10;  // 对预测结果进行缩放处理，乘以 10
    int n_ = testing_labels.rows();  // 获取测试数据的样本数

    ColVectorXd error;  // 存储误差
    prediction = round(prediction.array());  // 对预测结果进行四舍五入

    int num_correct = 0;  // 正确分类的样本数
    for (int i = 0; i < n_; i++){
        if(prediction[i] == testing_labels[i])  // 检查预测结果是否与测试标签一致
            num_correct++;  // 若一致，则正确分类数加一
    }
    double accuracy = num_correct/((double) n_);  // 计算准确率
    cout << "Accuracy on testing the trained model is " << accuracy * 100 << endl;  // 输出测试集上的准确率
}
