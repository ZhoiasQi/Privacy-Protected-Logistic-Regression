#include "sigmoid.hpp"

using namespace Eigen;
using Eigen::Matrix;
using namespace emp;
using namespace std;

//TODO: ʵ��MPC�Ѻõ�sigmoid����
ColVectorXi64 sigmoid(ColVectorXi64 x, int i_){
    //TODO: �Ȳ���ȫʵ��������
    int size = x.size();
    ColVectorXi64 res(size);

    // if(i_ == 0){
        for(int i = 0; i < size; i++){
            auto cur = x[i];
            if(cur > pow(2, 63)){
                uint64_t d = -1 * cur;
                int64_t a = d;
                if(a > 4096){
                    res[i] = 0;
                }
                else{
                    int64_t b = 4096 - a;
                    uint64_t c = b;
                    res[i] = c; 
                }
            }
            else{
                if(cur >= 4096){
                    res[i] = 8192;
                }
                else{
                    res[i] = x[i] + 4096;
                }
            }

        }

    return res;
    // }
    // else{
    //     for(int i = 0; i < size; i++){
    //         auto cur = -x[i];
    //         if(cur >= pow(2, 12)){
    //             res[i] = pow(2, 13);
    //         }
    //         else if(cur >= 0){
    //             res[i] = x[i] + pow(2, 12);
    //         }
    //         else{
    //             res[i] = pow(2, 13) - x[i];
    //         }
    //         res[i] = -res[i];
    //     }
    // }

    
}