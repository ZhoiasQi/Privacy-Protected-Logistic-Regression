# **Privacy Protected Logistic Regression**

*支持隐私保护的逻辑回归算法设计与实现*

# **Build**

*本课题使用了C++来构建项目，并且此项目需要在Ubuntu上运行。*

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
*下面两个程序分别用来测试直接使用逻辑回归理想情况下的训练准确率和支持隐私保护的逻辑回归算法的训练准确率用来比较*
* `ideal_functionality`
  - `./build/bin/ideal_functionality [num_iter]`
- `secure_ML`
  - On local machine
    - 在第一个终端输入
        - `./build/bin/PPLR 1 8000 [num_iter]` 
    - 在第二个终端输入
        - `./build/bin/PPLR 2 8000 [num_iter]`
