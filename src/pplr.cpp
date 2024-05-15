#include "read_WBDC.hpp"
#include "read_Arcene.hpp"
#include "preparation.hpp" 

using namespace Eigen;
using Eigen::Matrix;
using namespace emp;
using namespace std;

Traffic traffic;

IOFormat CommaInitFmt(StreamPrecision, DontAlignCols, ", ", ", ", "", "", " << ", ";");

//int NUM_INSTANCES = BATCH_SIZE;
int PARTY;

int main(int argc, char** argv){

    /****************************读取参数**************************************/
    int port, num_iters;
    string address;

    PARTY = atoi(argv[1]); 
    port = atoi(argv[2]);  
    //num_iters = atoi(argv[3]);  
    num_iters = 30;

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
    // vector<vector<uint64_t>> trainData = readData("../../arcene/ARCENE/arcene_train.data");
    // vector<vector<uint64_t>> testData = readData("../../arcene/ARCENE/arcene_valid.data");

    // vector<uint64_t> trainLabel = readLabel("../../arcene/ARCENE/arcene_train.labels");
    // vector<uint64_t> testLabel = readLabel("../../arcene/arcene_valid.labels");

    auto start = std::chrono::high_resolution_clock::now();

    // /****************************读取数据集**************************************/

    vector<BreastCancerInstance> dataSet;
    string fileName = "../../Dataset/wdbc.data";

    dataSet = read_WBDC_data(fileName);

    //print_WBDC_Data(dataSet);

    vector<vector<double>> dataFeatures;
    vector<double> dataLabels;

    dataFeatures = reverse_BreastCancerInstance_to_features(dataSet);
    dataLabels = reverse_BreastCancerInstance_to_labels(dataSet);

    //normalizeColumns(dataFeatures);
    // for(int i = 0; i < dataFeatures.size(); i++){
    //     for(int j = 0; j < dataFeatures.at(0).size(); j++){
    //         cout << dataFeatures.at(i).at(j) << " ";
    //     }
    //     cout << endl;
    // }
    
    /****************************对数据进行前期预处理**************************************/
    vector<vector<uint64_t>> uint64_dataFeatures;
    vector<uint64_t> uint64_dataLabels;

    uint64_dataFeatures = Fixed_Point_Representation_Features(dataFeatures);
    uint64_dataLabels = Fixed_Point_Representation_Labels(dataLabels);

    vector<vector<uint64_t>> training_Features;
    vector<vector<uint64_t>> testing_Features;
    vector<uint64_t> training_Labels;
    vector<uint64_t> testing_Labels;

    size_t trainingSize = 400;
    size_t offset = 169;

    training_Features.assign(uint64_dataFeatures.begin(), uint64_dataFeatures.begin() + trainingSize);
    testing_Features.assign(uint64_dataFeatures.begin() + trainingSize, uint64_dataFeatures.end());
    training_Labels.assign(uint64_dataLabels.begin(), uint64_dataLabels.begin() + trainingSize);
    testing_Labels.assign(uint64_dataLabels.begin() + trainingSize, uint64_dataLabels.end());

    size_t additionalCopies = num_iters - 1;

    if(additionalCopies > 0){
    
        vector<vector<uint64_t>> trainFeaturesCopy(training_Features);
        vector<uint64_t> trainLabelsCopy(training_Labels);

        for(size_t i = 0; i < additionalCopies; i++){

            vector<vector<uint64_t>> temp1(trainFeaturesCopy);
            vector<uint64_t> temp2(trainLabelsCopy);

            random_device rd;
            mt19937 gen(rd());

            vector<int> indices(temp1.size());
            iota(indices.begin(), indices.end(), 0);
            shuffle(indices.begin(), indices.end(), gen);

            vector<vector<uint64_t>> shuffled_temp1(temp1.size());
            vector<uint64_t> shuffled_temp2(temp2.size());
            for (int i = 0; i < temp1.size(); ++i) {
                shuffled_temp1[i] = temp1[indices[i]];
                shuffled_temp2[i] = temp2[indices[i]];
            }


            training_Features.insert(training_Features.end(), trainFeaturesCopy.begin(), trainFeaturesCopy.end());
            training_Labels.insert(training_Labels.end(), trainLabelsCopy.begin(), trainLabelsCopy.end());

        }
    }
    
    /****************************训练**************************************/
    cout << endl;
    cout << "========" << endl;
    cout << "Training" << endl;
    cout << "========" << endl;

    // cout << "Model Explanation: " << endl 
    //     << "Alice has all the data, but does not have enough arithmetic power to build the model," << endl
    //     <<  "so she use the auxiliary server Bob together to build the model, " << endl 
    //     << "and finally only Alice can get the machine learning model." << endl;

    // cout << "========" << endl;

    // cout << "Output: " << endl;

    TrainingParams params;

    traffic.offline = 0;
    traffic.online = 0;

    params.n = training_Features.size();
    cout << "Number of Instances: " << params.n << endl;
    params.d = training_Features[0].size();
    cout << "Number of Features: " << params.d << endl;

    RowMatrixXi64 X(params.n, params.d);
    ColVectorXi64 Y(params.n);

    vector2d_to_RowMatrixXi64(training_Features, X);
    vector_to_ColVectorXi64(training_Labels, Y);

    LogisticRegression trainModel(X, Y, params, io);

    ColVectorXi64 w;
    w.resize(params.d);
    
    if(PARTY == ALICE){
        w = trainModel.w;
        cout << "Alice has already got the model w" << endl;
        //cout << "w : " << endl << w << endl;
    }

    cout << "Offline Traffic = " << traffic.offline << "MB" << endl;
    cout << "Online Traffic = " << traffic.online << "MB" << endl;
    cout << "Total Traffic = " << traffic.online + traffic.offline << "MB" << endl;

    /****************************测试**************************************/
    cout << endl;
    cout << "=======" << endl;
    cout << "Testing" << endl;
    cout << "=======" << endl;

    // cout << "Model Explanation: " << endl 
    //     << "Alice has already obtained the model w in the training phase above, " << endl
    //     << "and Carol needs to use Alice's model for testing. " << endl
    //     << "But Alice doesn't want to reveal her model, and Carol doesn't want to reveal her data." << endl
    //     << "For the sake of code, since Bob is no longer needed, we continue to use terminal 2 as Carol for testing." << endl;

    // cout << "=======" << endl;

    // cout << "Output: " << endl;

    int n_= testing_Features.size();

    TestingParams t_params;

    t_params.n = n_;
    t_params.d = params.d;

    cout << "Number of Instances: " << t_params.n << endl;

    // if(PARTY == ALICE){
    //     offlineTest(testing_Features, testing_Labels, t_params, trainModel);
    // }
    
    RowMatrixXi64 testX(t_params.n, t_params.d);
    vector2d_to_RowMatrixXi64(testing_Features, testX); 

    ColVectorXi64 testY(t_params.n);  
    vector_to_ColVectorXi64(testing_Labels, testY);  

    TestLogisticRegression testModel(testX, testY, t_params, io);

    if(PARTY == ALICE){
        testModel.getW(w);
    }
    else{
        for(int i = 0; i < testModel.w.size(); i++){
            w[i] = 0;
        }
    }

    testModel.secret_share_w();

    auto acc = testModel.testmodel();

    if(PARTY == CAROL){
        cout << "Accuracy on testing the trained model is " << std::fixed << std::setprecision(2) << acc * 100 << "%" << endl;
    }

    auto end = std::chrono::high_resolution_clock::now(); 

    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);  

    std::cout << "Time: " << duration.count() << "s" << std::endl;  

    return 0;
}
