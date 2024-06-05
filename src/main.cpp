#include "tensor/tensor.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

namespace py = pybind11;

PYBIND11_MODULE(heatflow, m) {
    m.def("add", &add);
    m.def("matmul", &matmul);
}
