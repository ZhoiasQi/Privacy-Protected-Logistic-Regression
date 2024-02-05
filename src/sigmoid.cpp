#include "sigmoid.hpp"

using namespace Eigen;
using Eigen::Matrix;
using namespace emp;
using namespace std;

//TODO: 实现MPC友好的sigmoid函数
ColVectorXi64 sigmoid(ColVectorXi64 x){
    //TODO: 先不安全实现先用着
    int size = x.size();
    ColVectorXi64 res(size);

    //0和0.5为分界，小于0用1减去f（-x），0-0.5用x+0.5，大于等于0.5直接取1
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