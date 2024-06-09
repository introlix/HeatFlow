#include <iostream>
#include <vector>
#include <Eigen/Dense>

using namespace Eigen;

Eigen::MatrixXd add(const Eigen::MatrixXd &a, const Eigen::MatrixXd &b);
Eigen::MatrixXd subtract(const Eigen::MatrixXd &a, const Eigen::MatrixXd &b);
Eigen::MatrixXd matmul(const Eigen::MatrixXd &a, const Eigen::MatrixXd &b);
Eigen::MatrixXd divide(const Eigen::MatrixXd &a, const Eigen::MatrixXd &b);