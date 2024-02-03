#ifndef SETUP_HPP
#define SETUP_HPP

#include "util.hpp"  // �������õĹ��ߺ���
#include <emp-ot/emp-ot.h>  // ����emp-ot��

/**
 * SetupPhase ������ִ��Э������ý׶β�����
 */
class SetupPhase{
public:
    int n, d, t;  // ���������������������Լ� Bi �� Ci �ľ���ά��
    int party;  // ��ǰ���뷽��ALICE��BOB��
    RowMatrixXi64 Ai;  // ���� Ai
    ColMatrixXi64 Bi;  // ���� Bi
    ColMatrixXi64 Ci;  // ���� Ci
    ColMatrixXi64 Bi_;  // ���� Bi_
    ColMatrixXi64 Ci_;  // ���� Ci_

    emp::NetIO* io;  // ���� I/O ����ָ��
    emp::SHOTExtension<emp::NetIO>* send_ot;  // OT ���Ͷ˶���ָ��
    emp::SHOTExtension<emp::NetIO>* recv_ot;  // OT ���ն˶���ָ��
    emp::PRG prg;  // α���������������

    /**
     * ���캯�� SetupPhase ���ڳ�ʼ������
     * @param n ��������
     * @param d ��������
     * @param t ���� Bi �� Ci ��ά��
     * @param io ���� I/O ����ָ��
     */
    SetupPhase(int n, int d, int t, emp::NetIO* io){
        this->n = n;
        this->d = d;
        this->t = t;
        this->io = io;
        this->send_ot = new emp::SHOTExtension<emp::NetIO>(io);  // ��ʼ�� OT ���Ͷ˶���
        this->recv_ot = new emp::SHOTExtension<emp::NetIO>(io);  // ��ʼ�� OT ���ն˶���
        this->party = PARTY;  // ���õ�ǰ���뷽

        Ai.resize(n, d);  // �������� Ai �Ĵ�С
        Bi.resize(d, t);  // �������� Bi �Ĵ�С
        Ci.resize(BATCH_SIZE, t);  // �������� Ci �Ĵ�С
        Bi_.resize(BATCH_SIZE, t);  // �������� Bi_ �Ĵ�С
        Ci_.resize(d, t);  // �������� Ci_ �Ĵ�С

        initialize_matrices();  // ��ʼ������
        std::cout << "Matrices Initialized" << std::endl;  // �����Ϣ�������ѳ�ʼ��
    }

    void initialize_matrices();  // ��ʼ������ķ���
    void generateMTs();  // ���� MTs �ķ���
    void secure_mult(int N, int D, vector<vector<uint64_t>>& a,
                     vector<uint64_t>& b, vector<uint64_t> &c);  // ��ȫ�˷��ķ���
    void getMTs(SetupTriples* triples);  // ��ȡ MTs �ķ���
    void verify();  // ��֤����
};

#endif // SETUP_HPP
