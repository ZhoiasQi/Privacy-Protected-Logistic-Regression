#ifndef ONLINE_HPP
#define ONLINE_HPP

#include "util.hpp"

struct TrainingParams {
    int n;  // 样本数
    int d;  // 特征数
    int alpha_inv = LEARNING_RATE_INV;  // 学习率的倒数
};

#endif 
