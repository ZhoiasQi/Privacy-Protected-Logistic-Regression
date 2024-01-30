#include "read_WBDC.hpp"
#include "preparation.hpp" 

// using namespace Eigen;
// using Eigen::Matrix;
// using namespace emp;
using namespace std;

// IOFormat CommaInitFmt(StreamPrecision, DontAlignCols, ", ", ", ", "", "", " << ", ";");

// int NUM_INSTANCES = BATCH_SIZE;
int PARTY;

int main(int argc, char** argv){

    /****************************读取参数**************************************/
    // int port, num_iters;
    // string address;

    // PARTY = atoi(argv[1]);  // 从命令行参数获取party编号
    // port = atoi(argv[2]);  // 从命令行参数获取端口号
    // num_iters = atoi(argv[3]);  // 从命令行参数获取迭代次数

    // try{
    //     int x = -1;
    //     if(argc <= 4)
    //         throw x;
    //     address = argv[4];
    // } catch(int x) {
    //     address = "127.0.0.1";
    // }

    /****************************读取数据集**************************************/
    //将全部数据作为结构体读入dataSet数组
    vector<BreastCancerInstance> dataSet;
    string fileName = "../../Dataset/wdbc.data";

    dataSet = read_WBDC_data(fileName);

    //测试代码：打印中间结果
    //print_WBDC_Data(dataSet);

    //将dataSet数组转化为一个特征值矩阵和一个标签值向量
    vector<vector<double>> dataFeatures;
    vector<double> dataLabels;

    dataFeatures = reverse_BreastCancerInstance_to_features(dataSet);
    dataLabels = reverse_BreastCancerInstance_to_labels(dataSet);

    /****************************对数据进行前期预处理**************************************/
    //浮点数转定点数
    vector<vector<uint64_t>> uint64_dataFeatures;
    vector<uint64_t> uint64_dataLabels;

    uint64_dataFeatures = Fixed_Point_Representation_Features(dataFeatures);
    uint64_dataLabels = Fixed_Point_Representation_Labels(dataLabels);

    //划分测试集和训练集
    vector<vector<uint64_t>> training_Features;
    vector<vector<uint64_t>> testing_Features;
    vector<uint64_t> training_Labels;
    vector<uint64_t> testing_Labels;

    // 计算训练集的大小
    size_t trainingSize = uint64_dataFeatures.size() * 2 / 3;

    // 根据计算结果划分数据到训练集和测试集
    training_Features.assign(uint64_dataFeatures.begin(), uint64_dataFeatures.begin() + trainingSize);
    testing_Features.assign(uint64_dataFeatures.begin() + trainingSize, uint64_dataFeatures.end());
    training_Labels.assign(uint64_dataLabels.begin(), uint64_dataLabels.begin() + trainingSize);
    testing_Labels.assign(uint64_dataLabels.begin() + trainingSize, uint64_dataLabels.end());

    

    // NUM_INSTANCES = NUM_INSTANCES * num_iters;

    // NetIO* io = new NetIO(PARTY == ALICE ? nullptr : address.c_str(), port);
    // TrainingParams params;

    return 0;
}
