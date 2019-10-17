cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_ops)

find_package(Torch REQUIRED)

add_executable(bert-cpp bert.cpp)
target_link_libraries(bert-cpp "${TORCH_LIBRARIES}")
set_property(TARGET bert-cpp PROPERTY CXX_STANDARD 11)