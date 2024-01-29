#ifndef LR_HPP
#define LR_HPP
#include "setup_phase.hpp"
#include "online_phase.hpp"

class LinearRegression {
public:
    emp::NetIO* io; // 网络输入输出对象指针
    int party; // 当前参与方（ALICE或BOB）
    int n, d, t; // 训练数据大小（样本数、特征数）、迭代次数
    RowMatrixXi64 X; // 训练数据矩阵
    ColVectorXi64 Y; // 训练标签向量
    ColVectorXi64 w; // 权重向量（整数类型）
    ColVectorXd w_d; // 权重向量（浮点数类型）
    SetupPhase* setup; // 设置阶段对象指针
    OnlinePhase* online; // 在线阶段对象指针

    LinearRegression(RowMatrixXi64& training_data, ColVectorXi64& training_labels,
                     TrainingParams params, emp::NetIO* io) {
        this->n = params.n; // 样本数
        this->d = params.d; // 特征数
        this->t = (params.n) / BATCH_SIZE; // 迭代次数
        this->X = training_data; // 输入训练数据
        this->Y = training_labels; // 输入训练标签
        this->io = io; // 输入输出对象指针
        this->party = PARTY; // 参与方（ALICE或BOB）
        this->w.resize(d); // 初始化权重向量（整数类型）
        this->w_d.resize(d); // 初始化权重向量（浮点数类型）

        this->setup = new SetupPhase(n, d, t, io); // 创建设置阶段对象
        setup->generateMTs(); // 生成伪随机数
        std::cout << "Setup done" << std::endl;
        
        SetupTriples triples;
        setup->getMTs(&triples); // 获取伪随机数

        RowMatrixXi64 Xi(X.rows(), X.cols()); // 用于存储加噪后的训练数据矩阵
        ColVectorXi64 Yi(Y.rows(), Y.cols()); // 用于存储加噪后的训练标签向量
        
        if (party == emp::ALICE) { // 如果当前参与方是ALICE
            emp::PRG prg;
            RowMatrixXi64 rX(X.rows(), X.cols()); // 随机数矩阵（用于加噪）
            ColVectorXi64 rY(Y.rows(), Y.cols()); // 随机数向量（用于加噪）
            prg.random_data(rX.data(), X.rows() * X.cols() * sizeof(uint64_t)); // 生成随机数 rX
            prg.random_data(rY.data(), Y.rows() * Y.cols() * sizeof(uint64_t)); // 生成随机数 rY
            Xi = X + rX; // 加噪后的训练数据矩阵
            Yi = Y + rY; // 加噪后的训练标签向量
            rX *= -1; // 将 rX 中的元素取负
            rY *= -1; // 将 rY 中的元素取负
            send<RowMatrixXi64>(io, rX); // 将随机数 rX 发送给对方
            send<ColVectorXi64>(io, rY); // 将随机数 rY 发送给对方
        } else { // 如果当前参与方是BOB
            recv<RowMatrixXi64>(io, Xi); // 从对方接收加噪后的训练数据矩阵
            recv<ColVectorXi64>(io, Yi); // 从对方接收加噪后的训练标签向量
        }

        this->online = new OnlinePhase(params, io, &triples); // 创建在线阶段对象
        online->initialize(Xi, Yi); // 初始化在线阶段

        train_model(); // 训练模型
    }

    void train_model(); // 训练模型的函数
    void test_model(RowMatrixXd& testing_data, ColVectorXd& testing_labels); // 测试模型的函数
};
#endif
