cmake_minimum_required(VERSION 3.5)

project(ML LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# utility
add_library(utility
    ${PROJECT_SOURCE_DIR}/utility/utility.h
    ${PROJECT_SOURCE_DIR}/utility/utility.cpp)
# bayes
add_library(bayes
    ${PROJECT_SOURCE_DIR}/bayes/bayes.h
    ${PROJECT_SOURCE_DIR}/bayes/bayes.cpp)
# bpnn
add_library(bpnn
    ${PROJECT_SOURCE_DIR}/bpnn/bpnn.h
    ${PROJECT_SOURCE_DIR}/bpnn/bpnn.cpp)
# Linear Model
add_library(linear_model
    ${PROJECT_SOURCE_DIR}/linear_model/linear_model.h
    ${PROJECT_SOURCE_DIR}/linear_model/linear_model.cpp
    ${PROJECT_SOURCE_DIR}/linear_model/perceptron.h
    ${PROJECT_SOURCE_DIR}/linear_model/perceptron.cpp)
# rl
add_library(rl
    ${PROJECT_SOURCE_DIR}/RL/rl_basic.h
    ${PROJECT_SOURCE_DIR}/RL/dpg.h
    ${PROJECT_SOURCE_DIR}/RL/dpg.cpp
    ${PROJECT_SOURCE_DIR}/RL/dqn.h
    ${PROJECT_SOURCE_DIR}/RL/dqn.cpp
    ${PROJECT_SOURCE_DIR}/RL/ppo.h
    ${PROJECT_SOURCE_DIR}/RL/ppo.cpp)
# svm
add_library(svm
    ${PROJECT_SOURCE_DIR}/svm/svm.h
    ${PROJECT_SOURCE_DIR}/svm/svm.cpp)

add_executable(main main.cpp)
