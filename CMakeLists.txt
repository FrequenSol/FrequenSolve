cmake_minimum_required(VERSION 3.15)
project(frequensolve)

find_package(pybind11 REQUIRED)
pybind11_add_module(_mesh src/frequensolve/mesh/mesh_bindings.cpp)
