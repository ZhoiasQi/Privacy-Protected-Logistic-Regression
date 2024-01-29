#include "read_MNIST.hpp"  // 包含所需函数或类的头文件
#include "linear_regression.hpp"

using namespace Eigen;
using Eigen::Matrix;
using namespace emp;
using namespace std;

IOFormat CommaInitFmt(StreamPrecision, DontAlignCols, ", ", ", ", "", "", " << ", ";");

int NUM_IMAGES = BATCH_SIZE;
int PARTY;

int main(int argc, char** argv){
    int port, num_iters;
    string address;

    PARTY = atoi(argv[1]);  // 从命令行参数获取派对编号（假设 0 代表 Alice，1 代表 Bob）
    port = atoi(argv[2]);  // 从命令行参数获取端口号
    num_iters = atoi(argv[3]);  // 从命令行参数获取迭代次数

    try{
        int x = -1;
        if(argc <= 4)
            throw x;
        address = argv[4];
    } catch(int x) {
        address = "127.0.0.1";
    }

    NUM_IMAGES *= num_iters;

    NetIO* io = new NetIO(PARTY == ALICE ? nullptr : address.c_str(), port);

    TrainingParams params;

    cout << "========" << endl;
    cout << "Training" << endl;
    cout << "========" << endl;

    vector<vector<uint64_t> > training_data;  // 存储训练数据的向量
    vector<uint64_t> training_labels;  // 存储训练标签的向量

    read_MNIST_data<uint64_t>(true, training_data, params.n, params.d);  // 读取 MNIST 训练数据
    RowMatrixXi64 X(params.n, params.d);  // 定义训练数据矩阵
    vector2d_to_RowMatrixXi64(training_data, X);  // 将训练数据转化为矩阵形式
    X *= SCALING_FACTOR;  // 对训练数据进行缩放
    X /= 255;

    read_MNIST_labels<uint64_t>(true, training_labels);  // 读取 MNIST 训练标签
    ColVectorXi64 Y(params.n);  // 定义训练标签向量
    vector_to_ColVectorXi64(training_labels, Y);  // 将训练标签转化为向量形式
    Y *= SCALING_FACTOR;  // 对训练标签进行缩放
    Y /= 10;

    LinearRegression linear_regression(X, Y, params, io);  // 创建线性回归模型对象

    cout << "=======" << endl;
    cout << "Testing" << endl;
    cout << "=======" << endl;

    vector<double> testing_labels;  // 存储测试标签的向量
    int n_;

    vector<vector<double>> testing_data;  // 存储测试数据的向量
    read_MNIST_data<double>(false, testing_data, n_, params.d);  // 读取 MNIST 测试数据

    RowMatrixXd testX(n_, params.d);  // 定义测试数据矩阵
    vector2d_to_RowMatrixXd(testing_data, testX);  // 将测试数据转化为矩阵形式
    testX /= 255.0;  // 对测试数据进行缩放
    read_MNIST_labels<double>(false, testing_labels);  // 读取 MNIST 测试标签

    ColVectorXd testY(n_);  // 定义测试标签向量
    vector_to_ColVectorXd(testing_labels, testY);  // 将测试标签转化为向量形式
    linear_regression.test_model(testX, testY);  // 使用线性回归模型测试数据

    return 0;
}
