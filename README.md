# **Privacy Protected Logistic Regression**

*֧����˽�������߼��ع��㷨�����ʵ��*

# **Build**

*We use C++ to develop this project, and this work should work on Ubuntu.*

### **Preparation**

*��װemp-ot(v0.1�汾)*
```
//��װemp��׼������
git clone -b v0.1 https://github.com/emp-toolkit/emp-readme.git
cd emp-readme/
cd scripts/
bash install_packages.sh 

//��װemp-tool
git clone -b v0.1 https://github.com/emp-toolkit/emp-tool.git
cd emp-tool/
cmake . && sudo make install

//��װemp-ot
git clone -b v0.1 https://github.com/emp-toolkit/emp-ot.git
cd emp-ot/
cmake . && sudo make install
```

*��װEigen3��v3.3.7�汾��*
```
//���ز���ѹ
wget https://gitlab.com/libeigen/eigen/-/archive/3.3.7/eigen-3.3.7.tar.gz
tar -zxvf eigen-3.3.7.tar.gz

//��װ
cd eigen-3.3.7
mkdir build
cd build/
cmake ..
sudo make install
```

### **Installation**
```
git clone https://github.com/ZhoiasQi/Privacy-Protected-Logistic-Regression.git
cd Privacy-Protected-Logistic-Regression
mkdir build
cd build
cmake ..
make
```

### **Execution**
```
cd bin
./build/bin/result 1 8000 [num_iter] & ./build/bin/result 2 8000 [num_iter]
```
