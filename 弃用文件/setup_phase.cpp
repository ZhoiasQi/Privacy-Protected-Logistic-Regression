#include "setup_phase.hpp"

using namespace Eigen;
using Eigen::Matrix;
using namespace emp;
using namespace std;

/**
 * ��ʼ����������ݡ�
 * ʹ��α��������� prg ������� Ai��Bi �� Bi_ ���ݡ�
 */
void SetupPhase::initialize_matrices(){
    prg.random_data(Ai.data(), n * d * 8); // ������� Ai ����
    prg.random_data(Bi.data(), d * t * 8); // ������� Bi ����
    prg.random_data(Bi_.data(), BATCH_SIZE * t * 8); // ������� Bi_ ����
}

/**
 * ���� MTs��
 * ���� Ai��Bi �� Bi_ ���а�ȫ�˷����������� Ci �� Ci_��
 */
//�������ݾ��� Ai��Bi �� Bi_ ���а�ȫ�˷����������ɾ��� Ci �� Ci_�����У����� t ��ֵ�ֱ��ȡ Ai ���Ӿ��󣬽���ת�ò�����������ת��Ϊ��ά�� vector ���ݽṹ��Ȼ�󣬷ֱ�ִ�а�ȫ�˷��������õ� ci[i] �� ci_[i]����󣬽���Щ����ת��Ϊ��Ӧ�ľ������� Ci �� Ci_���ں���ĩβ�������Ϣ��ʾ MTs �����ɣ�������ѡ���� DEBUG ģʽ�½�����֤��
void SetupPhase::generateMTs(){
    vector<vector<uint64_t>> ci(t, vector<uint64_t>(BATCH_SIZE)); // Ci ����
    vector<vector<uint64_t>> ci_(t, vector<uint64_t>(d)); // Ci_ ����
    for(int i = 0; i < t; i++){
        RowMatrixXi64 Ai_b = Ai.block(i * BATCH_SIZE, 0, BATCH_SIZE, d); // ��ȡ Ai ���Ӿ��� Ai_b
        vector<vector<uint64_t>> ai(BATCH_SIZE, vector<uint64_t>(d)); 
        RowMatrixXi64_to_vector2d(Ai_b, ai); // �� Ai_b ת��Ϊ��ά vector
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

// ��δ����������ִ�а�ȫ�˷������ý׶Ρ�����������һ�´�����߼���

// - ���ȣ�����������һЩ��������`NUM_OT`���顢`total_ot`��`num_ot`�ȡ�
// - ��������ʹ��һ��ѭ�������ܵ�OT������ѭ����0��64������ÿ��ֵ������ʱ����`temp`����������洢��`NUM_OT[p]`�С�ͬʱ��ͨ����`N`���г�����ȡ�����㣬�������Ӧ��`NUM_OT`���ۼ�ÿ��`NUM_OT[p]`��ֵ��`total_ot`�У���������`D`��
// - Ȼ�󣬸���`total_ot`��ֵ��̬����������`block`���͵����飺`x0`��`x1`��`rec`�����ǵĴ�СΪ`total_ot`��
// - ��ʼ��һЩ������������`indexX0`��`indexX1`��`index_sigma`��
// - ����һ����������`bits_B`����СΪ64��������һ��ָ�򲼶������ָ��`sigma`����СΪ`total_ot`��
// - ����һ����άָ������`X0`����СΪ`N * BITLEN * D`��Ȼ��ʹ��ѭ��Ϊÿ��Ԫ�ظ�ֵ�������
// - ��������Ƕ�׵�ѭ�������ڴ���ÿ��`b`������Ԫ�غ�ÿ��OT��������
// - �ڲ�ѭ���У����Ƚ�`b[j]`ת��Ϊ��������`bits_B`��
// - Ȼ��ʹ��һЩλ�����ͼ����������������ʱ�������������Ǵ洢����Ӧ�������У�`X0`��`x0`��`x1`����
// - ��󣬽����ɵ�`x0`��`x1`�洢��`indexX0`��`indexX1`ָ����λ�ã�����Ӧ�ص�����Щ������

// ������������`party`�Ĳ�ͬ��ִ�в�ͬ�Ĳ�����
// - ���`party`Ϊ`ALICE`����ͨ�����ͺ�����`x0`��`x1`���͸��Է���
// - ���`party`Ϊ`BOB`����ͨ�����պ����������ԶԷ���`rec`��`sigma`��

// Ȼ�󣬸���`party`ִ�в�ͬ�Ĳ�����
// - ���`party`Ϊ`BOB`����ͨ�����ͺ����ٴν�`x0`��`x1`���͸��Է���
// - ���`party`Ϊ`ALICE`����ͨ�����պ����������ԶԷ���`rec`��`sigma`��

// �������Ĵ������Ҫ���ڼ���͸��½������`c`��
// - ����ѭ��Ƕ�׵�˳�����δ���`D`��Ԫ�صĳ˷���
// - ����ÿ��Ԫ�أ�ʹ��Ƕ�׵�ѭ�����������`z`ֵ��
// - ���ڲ�ѭ���У����ȸ���һЩ�����������Ҫ�����Ԫ��������������洢��`num_ot`�С�
// - ����������`rec`��������ȡ����Ӧ��Ԫ�أ�������ֳɸ�λ�͵�λ���֡�
// - ͨ��һϵ�е�λ���������㣬������õ��ĳ˻��ۼӵ��������`c`�С�
// - ��󣬽�`X0`�е�Ԫ�ؼ�ȥ�������`c`�Ķ�ӦԪ�ء�

// Ȼ��ͨ������ѭ���Խ������`c`ִ�м򵥵ĳ˷����㣬�ⲿ�ֿ����ǽ�����һ��δ���ܵĳ˷����㣬��ȷ�������׼ȷ�ԡ�

// ��󣬸��ݴ���ĺ������֣����ܻ���һЩ���������ڴ��ͷŲ�����

// ��ע�⣬��δ���Ľ��Ϳ��ܲ�������ĳЩϸ�ڿ�����Ҫ���������ĺ�����������ʵ�ֽ����Ʋ⡣
void SetupPhase::secure_mult(int N, int D, vector<vector<uint64_t>>& a,
                             vector<uint64_t>& b, vector<uint64_t> &c){
    int NUM_OT[BITLEN];
    int total_ot = 0, num_ot;

    // Calculate total number of OT
    for (int p = 0; p < 64; p++){
        int temp = 128/(64-p);
        NUM_OT[p] = N/temp;
        if (N % temp)
            NUM_OT[p]++;
        total_ot += NUM_OT[p];
    }
    total_ot *= D;

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

            for (int y = 0; y < num_ot; y++){
                sigma[index_sigma++] = bits_B[z];
            }

            for(int y = 0; y < num_ot; y++){
                int flag = elements_in_block;
                uint64_t temp_lo=0, temp_hi=0;
                uint64_t r_temp_lo=0, r_temp_hi=0;
                int elements_in_temp = 64/(64-z);
                int left_bitsLo = (64 % ((64-z)*elements_in_temp));

                r_temp_lo = (X0[indexA][z][j] << z);
                r_temp_lo >>= z;
                temp_lo = (randomA[indexA++] << z);
                temp_lo >>= z;
                flag--;
                for (int p=1; p<elements_in_temp; p++){
                    if (indexA <= N-1 && flag){
                        uint64_t r_next_element = (X0[indexA][z][j] << z);
                        r_next_element >>= z;
                        r_next_element <<= ((64-z) * p);
                        r_temp_lo ^= r_next_element;
                        uint64_t next_element = (randomA[indexA++] << z);
                        next_element >>= z;
                        next_element <<= ((64-z) * p);
                        temp_lo ^= next_element;
                        flag--;
                    }
                    else
                        break;
                }

                if (left_bitsLo){
                    if (indexA <= N-1 && flag){
                        uint64_t r_split_element = (X0[indexA][z][j] << z);
                        r_split_element >>= z;
                        r_temp_lo ^= (r_split_element << (64-left_bitsLo));
                        r_temp_hi ^= (r_split_element >> left_bitsLo);
                        uint64_t split_element = (randomA[indexA++] << z);
                        split_element >>= z;
                        temp_lo ^= (split_element << (64-left_bitsLo));
                        temp_hi ^= (split_element >> left_bitsLo);
                        flag--;
                    }
                }

                for (int p=0; p<elements_in_temp; p++){
                    if (indexA <= N-1 && flag){
                        uint64_t r_next_element = (X0[indexA][z][j] << z);
                        r_next_element >>= z;
                        if (left_bitsLo)
                            r_next_element <<= (((64-z)*p)+(64-z-left_bitsLo));
                        else
                            r_next_element <<= ((64-z)*p);
                        r_temp_hi ^= r_next_element;
                        uint64_t next_element = (randomA[indexA++] << z);
                        next_element >>= z;
                        if (left_bitsLo)
                            next_element <<= (((64-z)*p)+(64-z-left_bitsLo));
                        else
                            next_element <<= ((64-z)*p);
                        temp_hi ^= next_element;
                        flag--;
                    }
                    else
                        break;
                }

                x0[indexX0++] = makeBlock(r_temp_hi, r_temp_lo);
                x1[indexX1++] = makeBlock(temp_hi, temp_lo);

            }
        }
    }
    if (party == ALICE){
        send_ot->send(x0, x1, total_ot);
    }
    else if (party == BOB){
        recv_ot->recv(rec, sigma, total_ot);
    }

    if (party == BOB){
        send_ot->send(x0, x1, total_ot);
    }
    else if (party == ALICE){
        recv_ot->recv(rec, sigma, total_ot);
    }

    int indexRec = 0;
    for (int j = 0; j < D; j++){
        for (int z = 0; z < 64; z++){
            int indexA = 0;
            num_ot = NUM_OT[z];
            int elements_in_block = 128/(64-z);

            for (int y = 0; y < num_ot; y++){
                int flag = elements_in_block;
                uint64_t temp_lo = extract_lo64(rec[indexRec]);
                uint64_t temp_hi = extract_hi64(rec[indexRec++]);

                int elements_in_temp = 64/(64-z);
                int left_bitsLo = (64 % ((64-z) * elements_in_temp));
                uint64_t mask;
                if((64 - z) < 64)
                    mask = ((1ULL << (64-z)) - 1);
                else
                    mask = -1;

                for(int p = 0; p < elements_in_temp; p++){
                    if (indexA <= N-1 && flag) {
                        uint64_t next_element = (temp_lo & mask);
                        next_element <<= z;
                        c[indexA++] += next_element;
                        temp_lo >>= 64-z;
                        flag--;

                    }
                    else
                        break;
                }
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
                for(int p = 0; p < elements_in_temp; p++){
                    if (indexA <= N-1 && flag) {
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

    for(int i = 0; i < N; i++){
        for(int k = 0; k < D; k++){
            c[i] += (a[i][k] * b[k]);
        }
    }
}

void SetupPhase::getMTs(SetupTriples *triples){
    triples->Ai = this->Ai;
    triples->Bi = this->Bi;
    triples->Ci = this->Ci;
    triples->Bi_ = this->Bi_;
    triples->Ci_ = this->Ci_;
}

// ��δ�����һ��������֤�ĺ���`verify()`������`party`��ֵ��ͬ���ֱ�ִ�в�ͬ�Ĵ����߼����������н��ͣ�

// - `if (party == ALICE)` ��ʾ�����ǰ��`party`��`ALICE`����ִ�����´���飬����ִ��`else`�еĴ���顣
// - ��`if`���У����ȶ�����һЩ�ֲ�������������һЩ`Batched Integer Matrix`�Ķ������ڽ��պͼ������ݡ�
// - Ȼ��ʹ��`for`ѭ��������`t`�Σ����δ���ÿ��������
// - ��ѭ�����ڲ������ȸ���`i`��ֵ����`Ai`������ض����鸳ֵ��`Ai_b`����
// - Ȼ��һЩ����������ֵ����Ӧ�ľֲ�������
// - ���ʹ��`send(io, ...)`��������Щ�ֲ�������ֵ���͸�`io`���󣨿�����������������׽��֣���
// - `else`���е��߼���`if`�������ƣ����ǽ��������ݵĲ����ͼ������������һ��ѭ���С�
// - ��ÿ��ѭ���У�ͨ��`recv(io, ...)`��������`io`�����з��͵����ݣ������丳ֵ����Ӧ�ľֲ�������
// - Ȼ��ͨ��һϵ�еļ���������ֲ�������ֵ���и��¡�
// - ���ͨ���Ƚ�`C`��`AB`�Լ�`C_`��`AB_`��ֵ���ж���֤�Ƿ�ɹ��������֤�ɹ��������"Verification Successful"���������"Verification Failed"��

// ��δ�����ݲ�ͬ��`party`ִֵ�в�ͬ���߼���������֤һЩ���ݵ�һ���Ի���ȷ�ԡ�
void SetupPhase::verify(){
    if (party == ALICE) {
        RowMatrixXi64 Ai_b(BATCH_SIZE, d);
        ColMatrixXi64 Bi_b(d, 1); 
        ColMatrixXi64 Ci_b(BATCH_SIZE, 1);
        ColMatrixXi64 Bi_b_(BATCH_SIZE, 1);
        ColMatrixXi64 Ci_b_(d, 1);
        for(int i = 0; i < t; i++) {
            Ai_b = Ai.block((i * BATCH_SIZE) % n, 0, BATCH_SIZE, d);
            Bi_b = Bi.col(i);
            Ci_b = Ci.col(i);
            Bi_b_ = Bi_.col(i);
            Ci_b_ = Ci_.col(i);
            send(io, Ai_b);
            send(io, Bi_b);
            send(io, Ci_b);
            send(io, Bi_b_);
            send(io, Ci_b_);
        }
    }

    else {
        bool flag = true;
        bool flag_ = true;
        RowMatrixXi64 A(BATCH_SIZE, d);
        ColMatrixXi64 B(d, 1);
        ColMatrixXi64 C(BATCH_SIZE, 1);
        ColMatrixXi64 AB(BATCH_SIZE, 1);
        RowMatrixXi64 A_t(d, BATCH_SIZE);
        ColMatrixXi64 B_(BATCH_SIZE, 1);
        ColMatrixXi64 C_(d, 1);
        ColMatrixXi64 AB_(d, 1);
        for(int i = 0; i < t; i++) {
            RowMatrixXi64 Ai_b = Ai.block((i * BATCH_SIZE) % n, 0, BATCH_SIZE, d);
            ColMatrixXi64 Bi_b = Bi.col(i);
            ColMatrixXi64 Ci_b = Ci.col(i);
            ColMatrixXi64 Bi_b_ = Bi_.col(i);
            ColMatrixXi64 Ci_b_ = Ci_.col(i);

            recv(io, A);
            recv(io, B);
            recv(io, C);
            recv(io, B_);
            recv(io, C_);

            A += Ai_b;
            A_t = A.transpose();
            B += Bi_b;
            C += Ci_b;
            B_ += Bi_b_;
            C_ += Ci_b_;

            AB = A * B;
            AB_ = A_t * B_;

            if (C != AB) {
                flag_ = false;
                break;
            }

            if (C_ != AB_) {
                flag_ = false;
                break;
            }
        }

        if(flag == true) {
            cout << "Verification Successful" << endl;
        } else {
            cout << "Verification Failed" << endl;
        }

        if(flag_ == true) {
            cout << "Verification Successful" << endl;
        } else {
            cout << "Verification Failed" << endl;
        }
    }
}
