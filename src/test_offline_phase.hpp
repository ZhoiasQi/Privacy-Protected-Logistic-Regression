#ifndef TEST_OFFLINE_PHASE_HPP
#define TEST_OFFLINE_PHASE_HPP

#include <emp-ot/emp-ot.h>
#include "util.hpp"

using namespace emp;
using namespace std;

class TestSetUp{
    public:
    int n, d, t;
    int party;
    RowMatrixXi64 Ai;  // ���� Ai
    ColMatrixXi64 Bi;  // ���� Bi
    ColMatrixXi64 Ci;  // ���� Ci
    ColMatrixXi64 Bi_;  // ���� Bi_
    ColMatrixXi64 Ci_;  // ���� Ci_

    NetIO* io;  // ���� I/O ����ָ��
    SHOTExtension<emp::NetIO>* send_ot;  // OT ���Ͷ˶���ָ��
    SHOTExtension<emp::NetIO>* recv_ot;  // OT ���ն˶���ָ��
    PRG prg;  // α���������������

    TestSetUp(int n, int d, int t, NetIO* io){
        this->n = n;
        this->d = d;
        this->t = t;
        this->party = PARTY;
        this->io = io;
        this->send_ot = new SHOTExtension<NetIO>(io);
        this->recv_ot = new SHOTExtension<NetIO>(io);

        Ai.resize(n, d);  // �������� Ai �Ĵ�СΪn*d�ľ��󣬼�U
        Bi.resize(d, t);  // �������� Bi �Ĵ�СΪd*t�ľ�����Ϊ����ֵ��������Ϊ������������ѵ��,��V
        Ci.resize(n, t);  // �������� Ci �Ĵ�С��������ab�ĳ˻���Ϊ��������
        Bi_.resize(n, t);  // �������� Bi_ �Ĵ�С����V'
        Ci_.resize(d, t);  // �������� Ci_ �Ĵ�С��������UbT��V'�ĳ˻���Ϊ��������

        initialize_matrices();  // ��ʼ������
        cout << "Matrices Initialized" << endl;  
    }

    void initialize_matrices();
    void generateMTs();
    void secure_mult(int N, int D, vector<vector<uint64_t>>& a, vector<uint64_t>& b, vector<uint64_t> &c);
    void getMTs(SetupTriples *triples);
};

#endif