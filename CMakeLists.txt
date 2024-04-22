cmake_minimum_required(VERSION 3.5...3.27)
project(jax_apf LANGUAGES CXX)

add_subdirectory(extern/pybind11)

include_directories(${CMAKE_CURRENT_LIST_DIR}/src)
pybind11_add_module(cpu_ops ${CMAKE_CURRENT_LIST_DIR}/src/cpu_apf.cc)
install(TARGETS cpu_ops LIBRARY DESTINATION .)
