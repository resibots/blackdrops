#ifndef EIGEN_BINARY_MATRIX
#define EIGEN_BINARY_MATRIX

#include <fstream>
#include <cmath>

namespace Eigen {
    template <class Matrix>
    void write_binary(const std::string filename, const Matrix& matrix)
    {
        write_binary(filename.c_str(), matrix);
    }
    template <class Matrix>
    void write_binary(const char* filename, const Matrix& matrix)
    {
        std::ofstream out(filename, std::ios::out | std::ios::binary | std::ios::trunc);
        typename Matrix::Index rows = matrix.rows(), cols = matrix.cols();
        out.write((char*)(&rows), sizeof(typename Matrix::Index));
        out.write((char*)(&cols), sizeof(typename Matrix::Index));
        out.write((char*)matrix.data(), rows * cols * sizeof(typename Matrix::Scalar));
        out.close();
    }

    template <class Matrix>
    void read_binary(const std::string filename, Matrix& matrix)
    {
        read_binary(filename.c_str(), matrix);
    }

    template <class Matrix>
    void read_binary(const char* filename, Matrix& matrix)
    {
        std::ifstream in(filename, std::ios::in | std::ios::binary);
        typename Matrix::Index rows = 0, cols = 0;
        in.read((char*)(&rows), sizeof(typename Matrix::Index));
        in.read((char*)(&cols), sizeof(typename Matrix::Index));
        matrix.resize(rows, cols);
        in.read((char*)matrix.data(), rows * cols * sizeof(typename Matrix::Scalar));
        in.close();
    }

    MatrixXd colwise_sig(const MatrixXd& matrix)
    {
        VectorXd matrix_mean = matrix.colwise().mean();
        MatrixXd matrix_std = (matrix - matrix_mean.transpose().replicate(matrix.rows(), 1));
        matrix_std = matrix_std.array().square();
        MatrixXd matrix_sum = matrix_std.colwise().sum();
        matrix_sum *= (1.0 / double(matrix.rows() - 1));
        return matrix_sum.array().sqrt();
    }

    double percentile_v(const VectorXd& vector, int p)
    {
        VectorXd v = vector;
        std::sort(v.data(), v.data() + v.size());

        double pp = (p / 100.0) * (v.size() - 1.0);
        pp = std::round(pp * 1000.0) / 1000.0;
        int ind_below = std::floor(pp);
        int ind_above = std::ceil(pp);

        if (ind_below == ind_above)
            return v[ind_below];

        return v[ind_below] * (double(ind_above) - pp) + v[ind_above] * (pp - double(ind_below));
    }

    VectorXd percentile(const MatrixXd& matrix, int p)
    {
        VectorXd result(matrix.cols());
        for (int i = 0; i < matrix.cols(); i++) {
            result(i) = percentile_v(matrix.col(i), p);
        }
        return result;
    }
}
#endif
