#include "offline_phase.hpp"

using namespace Eigen;
using Eigen::Matrix;
using namespace emp;
using namespace std;

/**
 * ��ʼ����������ݡ�
 * ʹ��α��������� prg ������� Ai��Bi �� Bi_ ���ݡ�
 */
void OfflineSetUp::initialize_matrices(){
    prg.random_data(Ai.data(), n * d * 8); // ������� Ai ����
    prg.random_data(Bi.data(), d * t * 8); // ������� Bi ����
    prg.random_data(Bi_.data(), BATCH_SIZE * t * 8); // ������� Bi_ ����
}

/**
 * ���� MTs��
 * ���� Ai��Bi �� Bi_ ���а�ȫ�˷����������� Ci �� Ci_��
 * ���� t ��ֵ�ֱ��ȡ Ai ���Ӿ��󣬽���ת�ò�����������ת��Ϊ��ά�� vector ���ݽṹ��
 * �ֱ�ִ�а�ȫ�˷��������õ� ci[i] �� ci_[i]��
 * ����Щ����ת��Ϊ��Ӧ�ľ������� Ci �� Ci_���ں���ĩβ�������Ϣ��ʾ MTs �����ɣ�������ѡ���� DEBUG ģʽ�½�����֤��
 */
void OfflineSetUp::generateMTs(){
    vector<vector<uint64_t>> ci(t, vector<uint64_t>(BATCH_SIZE)); 
    vector<vector<uint64_t>> ci_(t, vector<uint64_t>(d)); 

    for(int i = 0; i < t; i++){
        RowMatrixXi64 Ai_b = Ai.block(i * BATCH_SIZE, 0, BATCH_SIZE, d); // ��ȡ Ai ���Ӿ��� Ai_b
        vector<vector<uint64_t>> ai(BATCH_SIZE, vector<uint64_t>(d)); 
        RowMatrixXi64_to_vector2d(Ai_b, ai); // ������ʽ�� Ai_b ת��Ϊ��ά vector ai

        RowMatrixXi64 Ai_bt = Ai_b.transpose(); // Ai_b ��ת�þ���
        vector<vector<uint64_t>> ai_t(d, vector<uint64_t>(BATCH_SIZE));
        RowMatrixXi64_to_vector2d(Ai_bt, ai_t); // �� Ai_bt ת��Ϊ��ά vector

        vector<uint64_t> bi = ColVectorXi64_to_vector(Bi.col(i)); // ��ȡ Bi ������������ת��Ϊ vector
        vector<uint64_t> bi_ = ColVectorXi64_to_vector(Bi_.col(i)); // ��ȡ Bi_ ������������ת��Ϊ vector

        secure_mult(BATCH_SIZE, d, ai, bi, ci[i]); // ִ�а�ȫ�˷��������õ� ci[i]
        secure_mult(d, BATCH_SIZE, ai_t, bi_, ci_[i]); // ִ�а�ȫ�˷��������õ� ci_[i]
    }

    vector2d_to_ColMatrixXi64(ci, Ci); // �� ci ת��Ϊ ColMatrixXi64 ���͵ľ��� Ci
    vector2d_to_ColMatrixXi64(ci_, Ci_); // �� ci_ ת��Ϊ ColMatrixXi64 ���͵ľ��� Ci_
    cout << "Triples Generated" << endl;

    #if DEBUG
        verify();
    #endif

}


void OfflineSetUp::secure_mult(int N, int D, vector<vector<uint64_t>>& a, vector<uint64_t>& b, vector<uint64_t> &c){
    // ��������
    int NUM_OT[BITLEN];
    int total_ot = 0, num_ot;

    // �����ܵ� OT ��Ŀ
    for (int p = 0; p < 64; p++){
        int temp = 128/(64-p);
        NUM_OT[p] = N/temp;
        if (N % temp)
            NUM_OT[p]++;
        total_ot += NUM_OT[p];
    }
    total_ot *= D;

    // ����洢�ռ�
    block *x0, *x1, *rec;
    x0 = new block[total_ot];
    x1 = new block[total_ot];
    rec = new block[total_ot];

    int indexX0 = 0;
    int indexX1 = 0;

    bool bits_B[64];
    bool* sigma;
    sigma = new bool[total_ot];
    int index_sigma = 0;

    uint64_t ***X0;
    X0 = new uint64_t**[N];
    for(int p = 0; p < N; p++) {
        X0[p] = new uint64_t*[BITLEN];
        for(int e = 0; e < BITLEN; e++){
            X0[p][e] = new uint64_t[D];
            prg.random_data(X0[p][e], D * 8);
        }
    }

    // ѭ�����а�ȫ�˷�Э��
    for(int j = 0; j < D; j++){
        int_to_bool(bits_B, b[j]);

        for (int z = 0; z < 64; z++){
            num_ot = NUM_OT[z];
            uint64_t randomA[N];

            for (int p = 0; p < N; p++){
                randomA[p] = X0[p][z][j] + a[p][j];
            }

            int elements_in_block = 128/(64-z);
            int indexA = 0;

            // ���������������Э����Ҫ��ֵ
            for (int y = 0; y < num_ot; y++){
                sigma[index_sigma++] = bits_B[z];
            }

            for(int y = 0; y < num_ot; y++){
                int flag = elements_in_block;
                uint64_t temp_lo=0, temp_hi=0;
                uint64_t r_temp_lo=0, r_temp_hi=0;
                int elements_in_temp = 64/(64-z);
                int left_bitsLo = (64 % ((64-z)*elements_in_temp));

                // ����Э������ֵ
                for (int p=1; p<elements_in_temp; p++){
                    //... ʡ�Բ��ִ���
                }

                // ���� x0 �� x1
                x0[indexX0++] = makeBlock(r_temp_hi, r_temp_lo);
                x1[indexX1++] = makeBlock(temp_hi, temp_lo);
            }
        }
    }

    // ���ͻ��������
    if (party == ALICE){
        send_ot->send(x0, x1, total_ot);
    }
    else if (party == BOB){
        recv_ot->recv(rec, sigma, total_ot);
    }

    if (party == BOB){
        // BOB���� x0 �� x1 ���Է�
        send_ot->send(x0, x1, total_ot);
    }
    else if (party == ALICE){
        // ALICE�������ݵ� rec �� sigma
        recv_ot->recv(rec, sigma, total_ot);
    }

int indexRec = 0;
for (int j = 0; j < D; j++) {
    // ���ѭ�������ڴ���ÿһ�� D ά���µ�����
    for (int z = 0; z < 64; z++) {
        // �ڲ�ѭ������ÿһ��λ�� z ���д���
        int indexA = 0;
        num_ot = NUM_OT[z]; // ��ȡ NUM_OT ����������Ϊ z ��ֵ
        int elements_in_block = 128 / (64 - z); // ÿ�������ж���Ԫ��

        for (int y = 0; y < num_ot; y++) {
            // ѭ������ num_ot ��
            int flag = elements_in_block; // ���ڿ���ÿ������Ԫ�صĴ������
            uint64_t temp_lo = extract_lo64(rec[indexRec]); // �� rec ����ȡ 64 λ�ĵ�λ����
            uint64_t temp_hi = extract_hi64(rec[indexRec++]); // �� rec ����ȡ 64 λ�ĸ�λ����

            int elements_in_temp = 64 / (64 - z); // ÿ��Ԫ����ռλ����
            int left_bitsLo = (64 % ((64 - z) * elements_in_temp)); // ����ʣ���λ��
            uint64_t mask;
            // ����λ�� z ���� mask
            if ((64 - z) < 64)
                mask = ((1ULL << (64 - z)) - 1);
            else
                mask = -1;

            for (int p = 0; p < elements_in_temp; p++) {
                // ����ÿ�����е�Ԫ��
                if (indexA <= N - 1 && flag) {
                    uint64_t next_element = (temp_lo & mask); // ��ȡ��һ��Ԫ��
                    next_element <<= z; // ���� z λ
                    c[indexA++] += next_element; // �ۼӵ� c ��
                    temp_lo >>= 64 - z; 
                    flag--; 

                } else
                    break;
            }
            if (left_bitsLo) {
                // �������ʣ���λ��
                if (indexA <= N - 1 && flag) {
                    uint64_t split_mask;
                    // ������ mask ������ʣ��λ��
                    if ((64 - z - left_bitsLo) < 64)
                        split_mask = ((1ULL << (64 - z - left_bitsLo)) - 1);
                    else
                        split_mask = -1;
                    uint64_t next_element = temp_hi & split_mask; // ����� mask ��ȡ����
                    next_element <<= left_bitsLo; 
                    next_element ^= temp_lo; 
                    next_element <<= z;
                    c[indexA++] += next_element; 
                    temp_hi >>= (64 - z - left_bitsLo); 
                    flag--; 
                }
            }
            for (int p = 0; p < elements_in_temp; p++) {
                // ����ʣ��ĸ�λ����
                if (indexA <= N - 1 && flag) {
                    uint64_t next_element = (temp_hi & mask); 
                    next_element <<= z; 
                    c[indexA++] += next_element; 
                    temp_hi >>= 64 - z; 

                    flag--;

                } else
                    break;
            }
        }
        for (int p = 0; p < N; p++) {
            // ���½������ c����ȥ X0 �ж�Ӧλ�õ�ֵ
            c[p] -= (X0[p][z][j] << z);
        }
    }
}

    // �ͷŶ�̬������ڴ�
    for(int p = 0; p < N; p++) {
        for(int e = 0; e < BITLEN; e++){
            delete X0[p][e];
        }
        delete X0[p];
    }
    delete X0;

    // �������յĽ���������Ҫ��Ҳ������ǰ���ѭ���м���
    for(int i = 0; i < N; i++){
        for(int k = 0; k < D; k++){
            c[i] += (a[i][k] * b[k]);
        }
    }
}

void OfflineSetUp::getMTs(SetupTriples *triples){
    triples->Ai = this->Ai;
    triples->Bi = this->Bi;
    triples->Ci = this->Ci;
    triples->Bi_ = this->Bi_;
    triples->Ci_ = this->Ci_;
}

void OfflineSetUp::verify(){

}