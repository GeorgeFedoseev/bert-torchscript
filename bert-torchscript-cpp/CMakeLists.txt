cmake_minimum_required(VERSION 3.12)
project(bert-cpp)

set(CMAKE_CXX_STANDARD 17)

set(Torch_DIR libtorch/share/cmake/Torch)
find_package(Torch REQUIRED)

add_executable(bert-cpp bert.cpp)
target_link_libraries(bert-cpp "${TORCH_LIBRARIES}")

set_property(TARGET bert-cpp PROPERTY CXX_STANDARD 11)