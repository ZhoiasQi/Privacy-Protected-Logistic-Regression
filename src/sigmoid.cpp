#include "sigmoid.hpp"

using namespace Eigen;
using Eigen::Matrix;
using namespace emp;
using namespace std;

//TODO: 实现MPC友好的sigmoid函数
ColVectorXi64 sigmoid(ColVectorXi64 x, int i_, emp::NetIO* io){

    //矩阵x表示前面的计算结果，i_表示是Alice还是Bob，Alice的i_ = 0, Bob的i_ = 1

    int size = x.size();

    for(int i = 0; i < size; i++){

        //Alice的时候
        if(i_ = 0){

            uint64_t u0 = x[i];
            uint64_t u1;

            send<uint64_t>(io, u0);
            recv<uint64_t>(io, u1);

            uint64_t temp = u0 + SCALING_FACTOR / 2 + u1;

            int b1;
            if(temp >> 63 == 1){
                b1 = 0;
            }
            else{
                b1 = 1;
            }

        }
        else{

        }

        


    }


    
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