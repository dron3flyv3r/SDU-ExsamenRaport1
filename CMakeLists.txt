cmake_minimum_required(VERSION 3.29)
project(ExsamenReport1)

find_package(OpenCV REQUIRED)

file(COPY "./imgs" DESTINATION ${CMAKE_BINARY_DIR})

set(CMAKE_CXX_STANDARD 20)

add_executable(ExsamenReport1 main.cpp)

target_link_libraries(ExsamenReport1 ${OpenCV_LIBS})