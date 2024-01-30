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

    /****************************��ȡ����**************************************/
    // int port, num_iters;
    // string address;

    // PARTY = atoi(argv[1]);  // �������в�����ȡparty���
    // port = atoi(argv[2]);  // �������в�����ȡ�˿ں�
    // num_iters = atoi(argv[3]);  // �������в�����ȡ��������

    // try{
    //     int x = -1;
    //     if(argc <= 4)
    //         throw x;
    //     address = argv[4];
    // } catch(int x) {
    //     address = "127.0.0.1";
    // }

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

    //���ֲ��Լ���ѵ����
    
    

    /****************************�����ݽ���ǰ��Ԥ����**************************************/
    // NUM_INSTANCES = NUM_INSTANCES * num_iters;

    // NetIO* io = new NetIO(PARTY == ALICE ? nullptr : address.c_str(), port);
    // TrainingParams params;

    return 0;
}
