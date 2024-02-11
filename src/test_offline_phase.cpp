#include "test_offline_phase.hpp"

using namespace Eigen;
using Eigen::Matrix;
using namespace emp;
using namespace std;

/**
 *  ��ʼ����������ݡ�
 *  ʹ��α��������� prg ������� Ai��Bi �� Bi_ ���ݡ�
 */
void TestSetUp::initialize_matrices(){
    prg.random_data(Ai.data(), n * d * 8); // ������� Ai ���ݣ���Ӧ�����е�U����ָ��x
    prg.random_data(Bi.data(), d * t * 8); // ������� Bi ���ݣ���Ӧ�����е�V����ÿһ��ָ��һ��w��һ�ε����е�ֵ
    prg.random_data(Bi_.data(), n * t * 8); // ������� Bi_ ���ݣ���Ӧ�����е�V'��ÿһ��ָ��Y*-Y��һ�ε����е�ֵ
}

/**
 *  �ָ�UV����ת�ø��ֱ仯
 *  ���õ�����Z�����γ���Ҫ�ĳ˷���Ԫ�� 
 */
void TestSetUp::generateMTs(){
    vector<vector<uint64_t>> ci(t, vector<uint64_t>(n)); 
    vector<vector<uint64_t>> ci_(t, vector<uint64_t>(d)); 

    for(int i = 0; i < t; i++){
        RowMatrixXi64 Ai_b = Ai.block(i * n, 0, n, d); // ��ȡ Ai ���Ӿ��� Ai_b
        vector<vector<uint64_t>> ai(n, vector<uint64_t>(d)); 
        RowMatrixXi64_to_vector2d(Ai_b, ai); // ������ʽ�� Ai_b ת��Ϊ��ά vector ai

        RowMatrixXi64 Ai_bt = Ai_b.transpose(); // Ai_b ��ת�þ���
        vector<vector<uint64_t>> ai_t(d, vector<uint64_t>(n));
        RowMatrixXi64_to_vector2d(Ai_bt, ai_t); // �� Ai_bt ת��Ϊ��ά vector

        vector<uint64_t> bi = ColVectorXi64_to_vector(Bi.col(i)); // ��ȡ Bi ������������ת��Ϊ vector
        vector<uint64_t> bi_ = ColVectorXi64_to_vector(Bi_.col(i)); // ��ȡ Bi_ ������������ת��Ϊ vector

        secure_mult(n, d, ai, bi, ci[i]); // ִ�а�ȫ�˷��������õ� ci[i]
        secure_mult(d, n, ai_t, bi_, ci_[i]); // ִ�а�ȫ�˷��������õ� ci_[i]
    }

    vector2d_to_ColMatrixXi64(ci, Ci); // �� ci ת��Ϊ ColMatrixXi64 ���͵ľ��� Ci
    vector2d_to_ColMatrixXi64(ci_, Ci_); // �� ci_ ת��Ϊ ColMatrixXi64 ���͵ľ��� Ci_
    cout << "Triples Generated" << endl;
}

/**
 *  @brief  ��ȫ�˷����������A��B�õ�C����Ӧ�������������������
 * 
 *  @param a ��һ������n*d����
 *  @param b �ڶ�������d*1����
 *  @param c ab�˻�����n*1����
 */
void TestSetUp::secure_mult(int N, int D, vector<vector<uint64_t>>& a, vector<uint64_t>& b, vector<uint64_t> &c){
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

    // ��ʼ������
    int indexX0 = 0;
    int indexX1 = 0;

    bool bits_B[64];
    bool* sigma;
    sigma = new bool[total_ot];
    int index_sigma = 0;

    // �����ڴ沢��������,�õ�N*64*D�ľ���
    uint64_t ***X0;
    X0 = new uint64_t**[N];
    for(int p = 0; p < N; p++) {
        X0[p] = new uint64_t*[BITLEN];
        for(int e = 0; e < BITLEN; e++){
            X0[p][e] = new uint64_t[D];
            prg.random_data(X0[p][e], D * 8);
        }
    }

    // ��������x0��x1
    for (int j = 0; j < D; j++) {  
        int_to_bool(bits_B, b[j]);  // ������b[j]ת��Ϊ��������bits_B

        for (int z = 0; z < 64; z++) {  
            num_ot = NUM_OT[z];  // ��ȡ��ǰ�ִε�OT�����洢��num_ot��
            uint64_t randomA[N];  // ��������ΪN�����������randomA

            for (int p = 0; p < N; p++) {
                randomA[p] = X0[p][z][j] + a[p][j];  // ���㲢�洢�����ֵ��randomA������
            }

            int elements_in_block = 128 / (64 - z);  // ����ÿ������Ԫ�ص�����
            int indexA = 0;

            for (int y = 0; y < num_ot; y++) {
                sigma[index_sigma++] = bits_B[z];  // ��bits_B[z]��ֵ�洢��sigma������
            }

            for(int y = 0; y < num_ot; y++){
                int flag = elements_in_block;  // ��Ǳ��������ڿ���ѭ���е�Ԫ������
                uint64_t temp_lo=0, temp_hi=0;  // ��ʱ�洢���������ڴ洢���
                uint64_t r_temp_lo=0, r_temp_hi=0;  // ��ʱ�洢���������ڴ洢���
                int elements_in_temp = 64/(64-z);  // ����ÿ��Ԫ�ذ�����λ��
                int left_bitsLo = (64 % ((64-z)*elements_in_temp));  // ����ʣ��λ��

                r_temp_lo = (X0[indexA][z][j] << z);  // �Ե�һ�������ݽ���λ�Ʋ���
                r_temp_lo >>= z;  // �ָ�ԭʼλ��
                temp_lo = (randomA[indexA++] << z);  // �Եڶ��������ݽ���λ�Ʋ���
                temp_lo >>= z;  // �ָ�ԭʼλ��
                flag--;

                for (int p=1; p<elements_in_temp; p++){
                    if (indexA <= N-1 && flag){  // ����Ƿ�����ʣ��Ԫ��
                        uint64_t r_next_element = (X0[indexA][z][j] << z);  // ��ȡ��һ������Ԫ�ز�����λ�Ʋ���
                        r_next_element >>= z;  // �ָ�ԭʼλ��
                        r_next_element <<= ((64-z) * p);  // ������λ��
                        r_temp_lo ^= r_next_element;  // �Խ������������
                        uint64_t next_element = (randomA[indexA++] << z);  // ��ȡ��һ������Ԫ�ز�����λ�Ʋ���
                        next_element >>= z;  // �ָ�ԭʼλ��
                        next_element <<= ((64-z) * p);  // ������λ��
                        temp_lo ^= next_element;  // �Խ������������
                        flag--;
                    }
                    else
                        break;
                }

                if (left_bitsLo){
                    if (indexA <= N-1 && flag){  // ����Ƿ�����ʣ��Ԫ��
                        uint64_t r_split_element = (X0[indexA][z][j] << z);  // ��ȡ��ֵ�����Ԫ�ز�����λ�Ʋ���
                        r_split_element >>= z;  // �ָ�ԭʼλ��
                        r_temp_lo ^= (r_split_element << (64-left_bitsLo));  // �Խ������������
                        r_temp_hi ^= (r_split_element >> left_bitsLo);  // �Խ������������
                        uint64_t split_element = (randomA[indexA++] << z);  // ��ȡ��ֵ�����Ԫ�ز�����λ�Ʋ���
                        split_element >>= z;  // �ָ�ԭʼλ��
                        temp_lo ^= (split_element << (64-left_bitsLo));  // �Խ������������
                        temp_hi ^= (split_element >> left_bitsLo);  // �Խ������������
                        flag--;
                    }
                }

                for (int p=0; p<elements_in_temp; p++){
                    if (indexA <= N-1 && flag){  // ����Ƿ�����ʣ��Ԫ��
                        uint64_t r_next_element = (X0[indexA][z][j] << z);  // ��ȡ��һ������Ԫ�ز�����λ�Ʋ���
                        r_next_element >>= z;  // �ָ�ԭʼλ��
                        if (left_bitsLo)
                            r_next_element <<= (((64-z)*p)+(64-z-left_bitsLo));  // ������λ��
                        else
                            r_next_element <<= ((64-z)*p);  // ������λ��
                        r_temp_hi ^= r_next_element;  // �Խ������������
                        uint64_t next_element = (randomA[indexA++] << z);  // ��ȡ��һ������Ԫ�ز�����λ�Ʋ���
                        next_element >>= z;  // �ָ�ԭʼλ��
                        if (left_bitsLo)
                            next_element <<= (((64-z)*p)+(64-z-left_bitsLo));  // ������λ��
                        else
                            next_element <<= ((64-z)*p);  // ������λ��
                        temp_hi ^= next_element;  // �Խ������������
                        flag--;
                    }
                    else
                        break;
                }

                x0[indexX0++] = makeBlock(r_temp_hi, r_temp_lo);  // ���ɿ�
                x1[indexX1++] = makeBlock(temp_hi, temp_lo);  // ���ɿ�
            }

        }
    }


    if (party == CAROL){
        send_ot->send(x0, x1, total_ot);
    }
    else if (party == ALICE){
        recv_ot->recv(rec, sigma, total_ot);
    }

    if (party == ALICE){
        send_ot->send(x0, x1, total_ot);
    }
    else if (party == CAROL){
        recv_ot->recv(rec, sigma, total_ot);
    }

    int indexRec = 0;
    // ѭ������ j �� 0 �� D
    for (int j = 0; j < D; j++){
        // ѭ������ z �� 0 �� 64
        for (int z = 0; z < 64; z++){
            int indexA = 0;
            num_ot = NUM_OT[z];
            // ����ÿ�����е�Ԫ������
            int elements_in_block = 128/(64-z);

            for (int y = 0; y < num_ot; y++){
                // ���Ԫ����Ŀ
                int flag = elements_in_block;
                // �ֱ���ȡ 64 λ��ĵ� 64 λ�͸� 64 λ
                uint64_t temp_lo = extract_lo64(rec[indexRec]);
                uint64_t temp_hi = extract_hi64(rec[indexRec++]);

                // ����ÿ������Ԫ�ص�����
                int elements_in_temp = 64/(64-z);
                // ����ʣ������λ��
                int left_bitsLo = (64 % ((64-z) * elements_in_temp));
                uint64_t mask;
                // ����������������
                if((64 - z) < 64)
                    mask = ((1ULL << (64-z)) - 1);
                else
                    mask = -1;

                // ����Ԫ�أ��������������Ԫ��
                for(int p = 0; p < elements_in_temp; p++){
                    if (indexA <= N-1 && flag) {
                        // ��ȡ��һ��Ԫ�أ���������뵽�������
                        uint64_t next_element = (temp_lo & mask);
                        next_element <<= z;
                        c[indexA++] += next_element;
                        temp_lo >>= 64-z;
                        flag--;

                    }
                    else
                        break;
                }
                // ����ʣ��ĵ�λ
                if (left_bitsLo){
                    if (indexA <= N-1 && flag){
                        uint64_t split_mask;
                        if((64-z-left_bitsLo) < 64)
                            split_mask = ((1ULL << (64-z-left_bitsLo)) -1);
                        else
                            split_mask = -1;
                        uint64_t next_element = temp_hi & split_mask;
                        next_element <<= left_bitsLo;
                        next_element ^= temp_lo;
                        next_element <<= z;
                        c[indexA++] += next_element;
                        temp_hi >>= (64-z-left_bitsLo);
                        flag--;
                    }
                }
                // ����Ԫ�أ�������������Ӹ�λԪ��
                for(int p = 0; p < elements_in_temp; p++){
                    if (indexA <= N-1 && flag) {
                        // ��ȡ��һ��Ԫ�أ���������뵽�������
                        uint64_t next_element = (temp_hi & mask);
                        next_element <<= z;
                        c[indexA++] += next_element;
                        temp_hi >>= 64-z;

                        flag--;
                    }
                    else
                        break;
                }
            }
            // ��ȥ X0 �����е�ֵ
            for (int p = 0; p < N; p++){
                c[p] -= (X0[p][z][j] << z);
            }
        }
    }


    for(int p = 0; p < N; p++) {
        for(int e = 0; e < BITLEN; e++){
            delete X0[p][e];
        }
        delete X0[p];
    }
    delete X0;

    //����C
    for(int i = 0; i < N; i++){
        for(int k = 0; k < D; k++){
            c[i] += (a[i][k] * b[k]);
        }
    }
}

//��ֵ������Ҫ�ĳ˷���Ԫ��
void TestSetUp::getMTs(SetupTriples *triples){
    triples->Ai = this->Ai;
    triples->Bi = this->Bi;
    triples->Ci = this->Ci;
    triples->Bi_ = this->Bi_;
    triples->Ci_ = this->Ci_;
}

