# **Privacy Protected Logistic Regression**

*支持隐私保护的逻辑回归算法设计与实现*

# **Build**

*We use C++ to develop this project, and this work should work on Ubuntu.*

### **Preparation**

*安装emp-ot(v0.1版本)*
```
//安装emp的准备工作
git clone -b v0.1 https://github.com/emp-toolkit/emp-readme.git
cd emp-readme/
cd scripts/
bash install_packages.sh 

//安装emp-tool
git clone -b v0.1 https://github.com/emp-toolkit/emp-tool.git
cd emp-tool/
cmake . && sudo make install

//安装emp-ot
git clone -b v0.1 https://github.com/emp-toolkit/emp-ot.git
cd emp-ot/
cmake . && sudo make install
```

*安装Eigen3（v3.3.7版本）*
```
//下载并解压
wget https://gitlab.com/libeigen/eigen/-/archive/3.3.7/eigen-3.3.7.tar.gz
tar -zxvf eigen-3.3.7.tar.gz

//安装
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
