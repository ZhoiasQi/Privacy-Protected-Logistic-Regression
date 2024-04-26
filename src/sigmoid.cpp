#include "sigmoid.hpp"
#include <emmintrin.h>

using namespace Eigen;
using Eigen::Matrix;
using namespace emp;
using namespace std;

void to_block(uint64_t value, emp::block & block_out) {  

    // 将 uint64_t 值复制到 emp::block 的开始位置  
    std::memcpy(&block_out[0], &value, sizeof(uint64_t));  

    // 如果 emp::block 比 uint64_t 大，则清零剩余部分（可选）  
    std::memset(&block_out[sizeof(uint64_t)], 0, sizeof(emp::block) - sizeof(uint64_t));  

}

uint64_t from_block(const emp::block & block_in) {  

    // 假设 emp::block 的前 8 字节包含我们要解码的 uint64_t 值  
    uint64_t value;  

    std::memcpy(&value, &block_in[0], sizeof(uint64_t));  

    return value;  
}

uint64_t msb(uint64_t value){
    int msb = 0; 
    for (int i = 63; i >= 0; --i) {  
        if ((value >> i) & 1) {  
            msb = i;  
            break;  
        }  
    }  
    return msb;  
}

typedef std::mt19937 RandomEngine;  

uint64_t generate_random_bit() {  
    static RandomEngine generator(std::random_device{}());  
    std::uniform_int_distribution<uint64_t> distribution(0, 1); 
    return distribution(generator);  
}  

ColVectorXi64 sigmoid(ColVectorXi64 x, int party, emp::NetIO* io){

    int size = x.size();

    vector<uint64_t> res;

    //随机数生成器
    PRG prg;
    //生成OT
    // Group* G = new Group;
    // OTCO<NetIO> ot(io, G);
    int length = 1;  //每次OT都只传一个block
    SHOTExtension<emp::NetIO>* send_ot_;  // OT 发送端对象指针
    SHOTExtension<emp::NetIO>* recv_ot_;  // OT 接收端对象指针

    send_ot_ = new SHOTExtension<NetIO>(io);
    recv_ot_ = new SHOTExtension<NetIO>(io);

    for(int i = 0; i < size; i++){

        //Alice
        if(party == 0){
            uint64_t b10, b20, b30, b40;

            uint64_t u0 = x[i];
        
            ColVectorXi64 v(1);
            recv<ColVectorXi64>(io, v);  
            auto vv = ColVectorXi64_to_vector(v);
            uint64_t u1 = vv.front(); 

            uint64_t b1 = msb(u0 + HALF_OF_ONE + u1);
            uint64_t b2 = msb(u0 - HALF_OF_ONE + u1);
            
            uint64_t b3 = 1 - b1;
            uint64_t b4 = b1 & (1 - b2);

            b30 = generate_random_bit();
            b40 = generate_random_bit();

            uint64_t b31 = b30 ^ b3;
            uint64_t b41 = b40 ^ b4;

            vector<uint64_t> bbb3;
            vector<uint64_t> bbb4;
            bbb3.push_back(b31);
            bbb4.push_back(b41);
            ColVectorXi64 vb1(1);
            ColVectorXi64 vb2(1);

            vector_to_ColVectorXi64(bbb3, vb1);  
            vector_to_ColVectorXi64(bbb4, vb2); 
            send<ColVectorXi64>(io, vb1);  
            send<ColVectorXi64>(io, vb2);  

            uint64_t m0, m1;
            block* temp0;
            block* temp1; 
            block* rec_temp;
            bool* choice;
            temp0 = new block[1];
            temp1 = new block[1];
            rec_temp = new block[1];
            choice = new bool[1];
            
            uint64_t r1;
            prg.random_data(&r1, sizeof(r1));
        
            m0 = b40 * u0 + r1;
            m1 = (1 - b40) * u0 + r1;

            temp0[0] = makeBlock(m0, m0);
            temp1[0] = makeBlock(m1, m1);

            send_ot_->send(temp0, temp1, length);

            uint64_t mb40;

            choice[0] = b41;

            recv_ot_->recv(rec_temp, choice, length);

            mb40 = extract_lo64(rec_temp[0]);

            //cout << mb40 << endl;

            uint64_t r3;
            prg.random_data(&r3, sizeof(r3));

            m0 = b30 + r3;
            m1 = (1 - b30) + r3;

            temp0[0] = makeBlock(m0, m0);
            temp1[0] = makeBlock(m1, m1);

            send_ot_->send(temp0, temp1, length);

            int64_t y0 = mb40 - r1 - r3;

            res.push_back(y0);

            delete[] temp0;
            delete[] temp1;
            delete[] rec_temp;
            delete[] choice;
        }

        //Bob
        else{
            uint64_t b11, b21, b31, b41;

            uint64_t u1 = x[i];

            vector<uint64_t> vec;
            vec.push_back(u1);
            ColVectorXi64 v(1);
            vector_to_ColVectorXi64(vec, v);  
            send<ColVectorXi64>(io, v);  

            ColVectorXi64 v3(1);
            ColVectorXi64 v4(1);
            recv<ColVectorXi64>(io, v3);
            recv<ColVectorXi64>(io, v4);

            auto vv3 = ColVectorXi64_to_vector(v3);
            auto vv4 = ColVectorXi64_to_vector(v4);

            b31 = vv3.front();
            b41 = vv4.front();

            uint64_t m0, m1;
            block* temp0;
            block* temp1; 
            block* rec_temp;
            bool* choice;
            temp0 = new block[1];
            temp1 = new block[1];
            rec_temp = new block[1];
            choice = new bool[1];

            uint64_t mb41;
            
            choice[0] = b41;

            recv_ot_->recv(rec_temp, choice, length);

            mb41 = extract_lo64(rec_temp[0]);
            
            uint64_t r2;
            prg.random_data(&r2, sizeof(r2));

            m0 = b41 * u1 + r2;
            m1 = (1 - b41) * u1 + r2;

            temp0[0] = makeBlock(m0, m0);
            temp1[0] = makeBlock(m1, m1);

            send_ot_->send(temp0, temp1, length);

            //第三轮OT
            uint64_t mb31;

            choice[0] = b31;

            recv_ot_->recv(rec_temp, choice, length);

            mb31 = extract_lo64(rec_temp[0]);

            int64_t y1 = mb41 + mb31 - r2;

            res.push_back(y1);

            delete[] temp0;
            delete[] temp1;
            delete[] rec_temp;
            delete[] choice;
        }

    }

    ColVectorXi64 result(size);

    vector_to_ColVectorXi64(res, result); 

    return result;

}

