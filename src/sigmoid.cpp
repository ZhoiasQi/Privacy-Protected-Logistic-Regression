#include "sigmoid.hpp"

using namespace Eigen;
using Eigen::Matrix;
using namespace emp;
using namespace std;

//TODO: ʵ��MPC�Ѻõ�sigmoid����
ColVectorXi64 sigmoid(ColVectorXi64 x){
    //TODO: �Ȳ���ȫʵ��������
    int size = x.size();
    ColVectorXi64 res(size);

    //0��0.5Ϊ�ֽ磬С��0��1��ȥf��-x����0-0.5��x+0.5�����ڵ���0.5ֱ��ȡ1
    for(int i = 0; i < size; i++){
        auto cur = x[i];
        if(cur >= pow(2, 12)){
            res[i] = pow(2, 13);
        }
        else if(cur >= 0){
            res[i] = x[i] + pow(2, 12);
        }
        else{
            res[i] = pow(2, 13) - x[i];
        }
    }

    return res;
}