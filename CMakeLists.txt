cmake_minimum_required(VERSION 3.0) #设置 CMake 的最低版本要求为 3.0

project(Privacy-Protected-Logistic-Regression)  #设置工程名为 Privacy-Protected-Logistic-Regression

set(NAME "Privacy-Protected-Logistic-Regression")  #设置变量 NAME 的值为 Privacy-Protected-Logistic-Regression

set (CMAKE_CXX_STANDARD 11)  #设置 C++ 标准为 C++11

set (CMAKE_BUILD_TYPE "Debug") # 设置编译类型为 Debug 模式

set (CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb") # 设置 Debug 模式下的 C++ 编译选项，包括关闭优化（-O0）、启用所有警告（-Wall）、生成调试信息（-g2 和 -ggdb）

set (CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall") # 设置 Release 模式下的 C++ 编译选项，包括启用高级优化（-O3）和启用所有警告（-Wall）

find_path(CMAKE_FOLDER NAMES cmake/emp-tool-config.cmake)  #在 CMAKE_FOLDER 中查找文件 cmake/emp-tool-config.cmake 的路径

include(${CMAKE_FOLDER}/cmake/common.cmake)  #包含 cmake/common.cmake 文件

include(${CMAKE_FOLDER}/cmake/source_of_randomness.cmake)  #包含 cmake/source_of_randomness.cmake 文件

include(${CMAKE_FOLDER}/cmake/threading.cmake)  #包含 cmake/threading.cmake 文件

find_package(emp-ot REQUIRED)  #查找并加载 emp-ot 包

find_package(Eigen3 REQUIRED NO_MODULE)  #查找并加载 Eigen3 包

INCLUDE_DIRECTORIES(${EMP-OT_INCLUDE_DIRS})  #添加引用 EMP-OT 包的头文件路径

add_executable(PPLR
    src/pplr.cpp
    src/read_WBDC.cpp
    src/preparation.cpp
    src/logistic_regression.cpp
    src/offline_phase.cpp 
    src/online_phase.cpp
    src/util.cpp
    src/sigmoid.cpp
    src/test_logistic_regression.cpp
    src/test_offline_phase.cpp
    src/test_online_phase.cpp
)

add_executable(ideal_functionality
    src/ideal_functionality.cpp 
    src/read_WBDC.cpp
    src/util.cpp
)

target_link_libraries(PPLR ${EMP-OT_LIBRARIES} Eigen3::Eigen)

target_link_libraries(ideal_functionality ${EMP-OT_LIBRARIES} Eigen3::Eigen)
