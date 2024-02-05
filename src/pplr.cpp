#include "read_WBDC.hpp"
#include "preparation.hpp" 

using namespace Eigen;
using Eigen::Matrix;
using namespace emp;
using namespace std;

IOFormat CommaInitFmt(StreamPrecision, DontAlignCols, ", ", ", ", "", "", " << ", ";");

//int NUM_INSTANCES = BATCH_SIZE;
int PARTY;

int main(int argc, char** argv){

    /****************************读取参数**************************************/
    int port, num_iters;
    string address;

    PARTY = atoi(argv[1]);  // 从命令行参数获取party编号
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

    NetIO* io = new NetIO(PARTY == ALICE ? nullptr : address.c_str(), port);

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

    // 划分测试集和训练集
    vector<vector<uint64_t>> training_Features;
    vector<vector<uint64_t>> testing_Features;
    vector<uint64_t> training_Labels;
    vector<uint64_t> testing_Labels;

    // 修改计算训练集的大小为128*3
    size_t trainingSize = 128 * 3;

    // 根据计算结果划分数据到训练集和测试集
    training_Features.assign(uint64_dataFeatures.begin(), uint64_dataFeatures.begin() + trainingSize);
    testing_Features.assign(uint64_dataFeatures.begin() + trainingSize, uint64_dataFeatures.end());
    training_Labels.assign(uint64_dataLabels.begin(), uint64_dataLabels.begin() + trainingSize);
    testing_Labels.assign(uint64_dataLabels.begin() + trainingSize, uint64_dataLabels.end());

    // 扩大训练集到num_iters倍用来反复利用
    size_t additionalCopies = num_iters - 1;

    if(additionalCopies > 0){
    
        //创建一个训练集的副本
        vector<vector<uint64_t>> trainFeaturesCopy(training_Features);
        vector<uint64_t> trainLabelsCopy(training_Labels);

        for(size_t i = 0; i < additionalCopies; i++){
            //复制训练集副本的副本
            vector<vector<uint64_t>> temp1(trainFeaturesCopy);
            vector<uint64_t> temp2(trainLabelsCopy);

            // 创建相同的随机数生成器
            random_device rd;
            mt19937 gen(rd());

            // 根据 temp1 的大小生成随机索引
            vector<int> indices(temp1.size());
            iota(indices.begin(), indices.end(), 0);
            shuffle(indices.begin(), indices.end(), gen);

            // 使用相同的索引顺序对 temp1 和 temp2 进行重排
            vector<vector<uint64_t>> shuffled_temp1(temp1.size());
            vector<uint64_t> shuffled_temp2(temp2.size());
            for (int i = 0; i < temp1.size(); ++i) {
                shuffled_temp1[i] = temp1[indices[i]];
                shuffled_temp2[i] = temp2[indices[i]];
            }

            // 将重排后的数据添加到原来的训练集后面
            training_Features.insert(training_Features.end(), shuffled_temp1.begin(), shuffled_temp1.end());
            training_Labels.insert(training_Labels.end(), shuffled_temp2.begin(), shuffled_temp2.end());

        }
    }
    
    /****************************训练**************************************/
    cout << "========" << endl;
    cout << "Training" << endl;
    cout << "========" << endl;

    //转成调用Eigen库的形式便于后续处理
    TrainingParams params;

    params.n = training_Features.size();
    cout << "Number of Instances: " << params.n << endl;
    params.d = training_Features[0].size();
    cout << "Number of Features: " << params.d << endl;

    RowMatrixXi64 X(params.n, params.d);
    ColVectorXi64 Y(params.n);

    vector2d_to_RowMatrixXi64(training_Features, X);
    vector_to_ColVectorXi64(training_Labels, Y);

    LogisticRegression logisticRegression(X, Y, params, io);

    /****************************测试**************************************/
    cout << "=======" << endl;
    cout << "Testing" << endl;
    cout << "=======" << endl;

    //因为前期不理解含义把所有的训练集测试集全都浮点数转定点数了，现在要把训练集的转回来
    vector<vector<double>> testFeaturesD;
    vector<double> testLabelsD;

    testFeaturesD = Fix_to_Double_F(testing_Features);
    testLabelsD = Fix_to_Double_L(testing_Labels);

    int n_= testFeaturesD.size();
    
    RowMatrixXd testX(n_, params.d);
    vector2d_to_RowMatrixXd(testFeaturesD, testX); 

    ColVectorXd testY(n_);  
    vector_to_ColVectorXd(testLabelsD, testY);  

    //logisticRegression.test_model(testX, testY); 

    return 0;
}
