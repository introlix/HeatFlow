#include "tensor.hpp"

Eigen::MatrixXd add(const Eigen::MatrixXd &a, const Eigen::MatrixXd &b) {
    return a + b;
}

Eigen::MatrixXd subtract(const Eigen::MatrixXd &a, const Eigen::MatrixXd &b) {
    return a - b;
}

Eigen::MatrixXd matmul(const Eigen::MatrixXd &a, const Eigen::MatrixXd &b) {
    return a * b;
}

Eigen::MatrixXd divide(const Eigen::MatrixXd &a, const Eigen::MatrixXd &b) {
    return a.array() / b.array();
}