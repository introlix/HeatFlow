#include "tensor/tensor.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

namespace py = pybind11;

PYBIND11_MODULE(heatflow_cpp, m) {
    m.def("add_cpp", &add);
    m.def("subtract_cpp", &subtract);
    m.def("matmul_cpp", &matmul);
    m.def("divide_cpp", &divide);
}
