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

    /****************************��ȡ����**************************************/
    int port, num_iters;
    string address;

    PARTY = atoi(argv[1]);  // �������в�����ȡparty���
    port = atoi(argv[2]);  // �������в�����ȡ�˿ں�
    num_iters = atoi(argv[3]);  // �������в�����ȡ��������

    try{
        int x = -1;
        if(argc <= 4)
            throw x;
        address = argv[4];
    } catch(int x) {
        address = "127.0.0.1";
    }

    NetIO* io = new NetIO(PARTY == ALICE ? nullptr : address.c_str(), port);

    /****************************��ȡ���ݼ�**************************************/
    //��ȫ��������Ϊ�ṹ�����dataSet����
    vector<BreastCancerInstance> dataSet;
    string fileName = "../../Dataset/wdbc.data";

    dataSet = read_WBDC_data(fileName);

    //���Դ��룺��ӡ�м���
    //print_WBDC_Data(dataSet);

    //��dataSet����ת��Ϊһ������ֵ�����һ����ǩֵ����
    vector<vector<double>> dataFeatures;
    vector<double> dataLabels;

    dataFeatures = reverse_BreastCancerInstance_to_features(dataSet);
    dataLabels = reverse_BreastCancerInstance_to_labels(dataSet);
    
    /****************************�����ݽ���ǰ��Ԥ����**************************************/
    //������ת������
    vector<vector<uint64_t>> uint64_dataFeatures;
    vector<uint64_t> uint64_dataLabels;

    uint64_dataFeatures = Fixed_Point_Representation_Features(dataFeatures);
    uint64_dataLabels = Fixed_Point_Representation_Labels(dataLabels);

    // ���ֲ��Լ���ѵ����
    vector<vector<uint64_t>> training_Features;
    vector<vector<uint64_t>> testing_Features;
    vector<uint64_t> training_Labels;
    vector<uint64_t> testing_Labels;

    // �޸ļ���ѵ�����Ĵ�СΪ128*3,�ĳ�32*10��
    size_t trainingSize = BATCH_SIZE * 10;

    // ���ݼ������������ݵ�ѵ�����Ͳ��Լ�
    training_Features.assign(uint64_dataFeatures.begin(), uint64_dataFeatures.begin() + trainingSize);
    testing_Features.assign(uint64_dataFeatures.begin() + trainingSize, uint64_dataFeatures.end());
    training_Labels.assign(uint64_dataLabels.begin(), uint64_dataLabels.begin() + trainingSize);
    testing_Labels.assign(uint64_dataLabels.begin() + trainingSize, uint64_dataLabels.end());

    // ����ѵ������num_iters��������������
    size_t additionalCopies = num_iters - 1;

    if(additionalCopies > 0){
    
        //����һ��ѵ�����ĸ���
        vector<vector<uint64_t>> trainFeaturesCopy(training_Features);
        vector<uint64_t> trainLabelsCopy(training_Labels);

        for(size_t i = 0; i < additionalCopies; i++){
            //����ѵ���������ĸ���
            vector<vector<uint64_t>> temp1(trainFeaturesCopy);
            vector<uint64_t> temp2(trainLabelsCopy);

            // ������ͬ�������������
            random_device rd;
            mt19937 gen(rd());

            // ���� temp1 �Ĵ�С�����������
            vector<int> indices(temp1.size());
            iota(indices.begin(), indices.end(), 0);
            shuffle(indices.begin(), indices.end(), gen);

            // ʹ����ͬ������˳��� temp1 �� temp2 ��������
            vector<vector<uint64_t>> shuffled_temp1(temp1.size());
            vector<uint64_t> shuffled_temp2(temp2.size());
            for (int i = 0; i < temp1.size(); ++i) {
                shuffled_temp1[i] = temp1[indices[i]];
                shuffled_temp2[i] = temp2[indices[i]];
            }

            // �����ź��������ӵ�ԭ����ѵ��������
            training_Features.insert(training_Features.end(), shuffled_temp1.begin(), shuffled_temp1.end());
            training_Labels.insert(training_Labels.end(), shuffled_temp2.begin(), shuffled_temp2.end());

        }
    }
    
    /****************************ѵ��**************************************/
    cout << endl;
    cout << "========" << endl;
    cout << "Training" << endl;
    cout << "========" << endl;

    cout << "Model Explanation: " << endl 
        << "Alice has all the data, but does not have enough arithmetic power to build the model," << endl
        <<  "so she use the auxiliary server Bob together to build the model, " << endl 
        << "and finally only Alice can get the machine learning model." << endl;

    cout << "========" << endl;

    cout << "Output: " << endl;

    //ת�ɵ���Eigen�����ʽ���ں�������
    TrainingParams params;

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

    /****************************����**************************************/
    cout << endl;
    cout << "=======" << endl;
    cout << "Testing" << endl;
    cout << "=======" << endl;

    cout << "Model Explanation: " << endl 
        << "Alice has already obtained the model w in the training phase above, " << endl
        << "and Carol needs to use Alice's model for testing. " << endl
        << "But Alice doesn't want to reveal her model, and Carol doesn't want to reveal her data." << endl
        << "For the sake of code, since Bob is no longer needed, we continue to use terminal 2 as Carol for testing." << endl;

    cout << "=======" << endl;

    cout << "Output: " << endl;

    int n_= testing_Features.size();

    TestingParams t_params;

    t_params.n = n_;
    t_params.d = params.d;

    cout << "Number of Instances: " << t_params.n << endl;
    
    RowMatrixXi64 testX(t_params.n, t_params.d);
    vector2d_to_RowMatrixXi64(testing_Features, testX); 

    ColVectorXi64 testY(t_params.n);  
    vector_to_ColVectorXi64(testing_Labels, testY);  

    TestLogisticRegression testModel(X, Y, t_params, io);

    if(PARTY == ALICE){
        testModel.getW(w);
    }
    else{
        for(int i = 0; i < testModel.w.size(); i++){
            w[i] = 0;
        }
    }

    testModel.secret_share_w();

    testModel.test_model();

    return 0;
}
