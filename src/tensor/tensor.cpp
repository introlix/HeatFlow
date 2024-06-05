#include "tensor.hpp"

double add(double a, double b) {
   return a + b;
}

Eigen::MatrixXd matmul(const Eigen::MatrixXd &a, const Eigen::MatrixXd &b) {
    return a * b;
}

int main() {
    Eigen::Matrix2d a;
    a << 1, 2,
        3, 4,
        4, 5,
        6, 7;

    Eigen::Matrix2d b;
    b << 1, 2, 3, 4,
        3, 4, 5, 6;

    std::cout << matmul(a, b) << std::endl;

    return 0;
}