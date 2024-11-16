cmake_minimum_required(VERSION 3.12)
project(DeepLearningDemo)

set(CMAKE_CXX_STANDARD 14)

set(SOURCES
	linear.cpp)

add_executable(linear ${SOURCES})
