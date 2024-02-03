#include "setup_phase.hpp"

using namespace Eigen;
using Eigen::Matrix;
using namespace emp;
using namespace std;

/**
 * 初始化矩阵的数据。
 * 使用伪随机生成器 prg 随机生成 Ai、Bi 和 Bi_ 数据。
 */
void SetupPhase::initialize_matrices(){
    prg.random_data(Ai.data(), n * d * 8); // 随机生成 Ai 数据
    prg.random_data(Bi.data(), d * t * 8); // 随机生成 Bi 数据
    prg.random_data(Bi_.data(), BATCH_SIZE * t * 8); // 随机生成 Bi_ 数据
}

/**
 * 生成 MTs。
 * 根据 Ai、Bi 和 Bi_ 进行安全乘法操作，生成 Ci 和 Ci_。
 */
//函数根据矩阵 Ai、Bi 和 Bi_ 进行安全乘法操作，生成矩阵 Ci 和 Ci_。其中，根据 t 的值分别截取 Ai 的子矩阵，进行转置操作，并将其转换为二维的 vector 数据结构。然后，分别执行安全乘法操作，得到 ci[i] 和 ci_[i]。最后，将这些数据转换为相应的矩阵类型 Ci 和 Ci_。在函数末尾，输出信息表示 MTs 已生成，并可以选择在 DEBUG 模式下进行验证。
void SetupPhase::generateMTs(){
    vector<vector<uint64_t>> ci(t, vector<uint64_t>(BATCH_SIZE)); // Ci 数据
    vector<vector<uint64_t>> ci_(t, vector<uint64_t>(d)); // Ci_ 数据
    for(int i = 0; i < t; i++){
        RowMatrixXi64 Ai_b = Ai.block(i * BATCH_SIZE, 0, BATCH_SIZE, d); // 截取 Ai 的子矩阵 Ai_b
        vector<vector<uint64_t>> ai(BATCH_SIZE, vector<uint64_t>(d)); 
        RowMatrixXi64_to_vector2d(Ai_b, ai); // 将 Ai_b 转换为二维 vector
        RowMatrixXi64 Ai_bt = Ai_b.transpose(); // Ai_b 的转置矩阵
        vector<vector<uint64_t>> ai_t(d, vector<uint64_t>(BATCH_SIZE));
        RowMatrixXi64_to_vector2d(Ai_bt, ai_t); // 将 Ai_bt 转换为二维 vector
        vector<uint64_t> bi = ColVectorXi64_to_vector(Bi.col(i)); // 获取 Bi 的列向量，并转换为 vector
        vector<uint64_t> bi_ = ColVectorXi64_to_vector(Bi_.col(i)); // 获取 Bi_ 的列向量，并转换为 vector
        secure_mult(BATCH_SIZE, d, ai, bi, ci[i]); // 执行安全乘法操作，得到 ci[i]
        secure_mult(d, BATCH_SIZE, ai_t, bi_, ci_[i]); // 执行安全乘法操作，得到 ci_[i]
    }
    vector2d_to_ColMatrixXi64(ci, Ci); // 将 ci 转换为 ColMatrixXi64 类型的矩阵 Ci
    vector2d_to_ColMatrixXi64(ci_, Ci_); // 将 ci_ 转换为 ColMatrixXi64 类型的矩阵 Ci_
    cout << "Triples Generated" << endl;
#if DEBUG
    verify();
#endif
}

// 这段代码的作用是执行安全乘法的设置阶段。让我来解释一下代码的逻辑：

// - 首先，代码声明了一些变量，如`NUM_OT`数组、`total_ot`、`num_ot`等。
// - 接下来，使用一个循环计算总的OT数量。循环从0到64，对于每个值计算临时变量`temp`，并将结果存储在`NUM_OT[p]`中。同时，通过对`N`进行除法和取余运算，计算出对应的`NUM_OT`。累加每个`NUM_OT[p]`的值到`total_ot`中，最后将其乘以`D`。
// - 然后，根据`total_ot`的值动态分配了三个`block`类型的数组：`x0`、`x1`和`rec`，它们的大小为`total_ot`。
// - 初始化一些索引变量，如`indexX0`、`indexX1`和`index_sigma`。
// - 创建一个布尔数组`bits_B`，大小为64，并创建一个指向布尔数组的指针`sigma`，大小为`total_ot`。
// - 分配一个三维指针数组`X0`，大小为`N * BITLEN * D`。然后，使用循环为每个元素赋值随机数。
// - 进入两个嵌套的循环，用于处理每个`b`向量的元素和每个OT的数量。
// - 内层循环中，首先将`b[j]`转换为布尔数组`bits_B`。
// - 然后使用一些位操作和计算生成随机数和临时变量，并将它们存储在相应的数组中（`X0`、`x0`和`x1`）。
// - 最后，将生成的`x0`和`x1`存储在`indexX0`和`indexX1`指定的位置，并相应地递增这些索引。

// 接下来，根据`party`的不同，执行不同的操作：
// - 如果`party`为`ALICE`，则通过发送函数将`x0`和`x1`发送给对方。
// - 如果`party`为`BOB`，则通过接收函数接收来自对方的`rec`和`sigma`。

// 然后，根据`party`执行不同的操作：
// - 如果`party`为`BOB`，则通过发送函数再次将`x0`和`x1`发送给对方。
// - 如果`party`为`ALICE`，则通过接收函数接收来自对方的`rec`和`sigma`。

// 接下来的代码段主要用于计算和更新结果数组`c`：
// - 根据循环嵌套的顺序，依次处理`D`个元素的乘法。
// - 对于每个元素，使用嵌套的循环来处理各个`z`值。
// - 在内层循环中，首先根据一些索引计算出需要处理的元素数量，并将其存储在`num_ot`中。
// - 接下来，从`rec`数组中提取出相应的元素，并将其分成高位和低位部分。
// - 通过一系列的位操作和运算，将计算得到的乘积累加到结果数组`c`中。
// - 最后，将`X0`中的元素减去结果数组`c`的对应元素。

// 然后，通过两个循环对结果数组`c`执行简单的乘法计算，这部分可能是进行了一次未加密的乘法计算，以确保结果的准确性。

// 最后，根据代码的后续部分，可能还有一些清理工作和内存释放操作。

// 请注意，这段代码的解释可能不完整，某些细节可能需要根据上下文和其他函数的实现进行推测。
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

// 这段代码是一个用于验证的函数`verify()`，根据`party`的值不同，分别执行不同的代码逻辑。让我逐行解释：

// - `if (party == ALICE)` 表示如果当前的`party`是`ALICE`，则执行以下代码块，否则执行`else`中的代码块。
// - 在`if`块中，首先定义了一些局部变量，包含了一些`Batched Integer Matrix`的对象，用于接收和计算数据。
// - 然后使用`for`循环，迭代`t`次，依次处理每个迭代。
// - 在循环体内部，首先根据`i`的值，将`Ai`对象的特定区块赋值给`Ai_b`对象。
// - 然后将一些列向量对象赋值给对应的局部变量。
// - 最后，使用`send(io, ...)`函数将这些局部变量的值发送给`io`对象（可能是输出流或网络套接字）。
// - `else`块中的逻辑和`if`块中类似，但是将接收数据的操作和计算操作放在了一个循环中。
// - 在每次循环中，通过`recv(io, ...)`函数接收`io`对象中发送的数据，并将其赋值给对应的局部变量。
// - 然后通过一系列的计算操作将局部变量的值进行更新。
// - 最后，通过比较`C`和`AB`以及`C_`和`AB_`的值来判断验证是否成功。如果验证成功，则输出"Verification Successful"；否则输出"Verification Failed"。

// 这段代码根据不同的`party`值执行不同的逻辑，用于验证一些数据的一致性或正确性。
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
