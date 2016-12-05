#ifndef EIGEN_BINARY_MATRIX
#define EIGEN_BINARY_MATRIX

#include <fstream>
#include <cmath>

namespace Eigen {
    template<class Matrix>
    void write_binary(const char* filename, const Matrix& matrix) {
        std::ofstream out(filename,std::ios::out | std::ios::binary | std::ios::trunc);
        typename Matrix::Index rows=matrix.rows(), cols=matrix.cols();
        out.write((char*) (&rows), sizeof(typename Matrix::Index));
        out.write((char*) (&cols), sizeof(typename Matrix::Index));
        out.write((char*) matrix.data(), rows*cols*sizeof(typename Matrix::Scalar) );
        out.close();
    }

    template<class Matrix>
    void read_binary(const char* filename, Matrix& matrix){
        std::ifstream in(filename,std::ios::in | std::ios::binary);
        typename Matrix::Index rows=0, cols=0;
        in.read((char*) (&rows),sizeof(typename Matrix::Index));
        in.read((char*) (&cols),sizeof(typename Matrix::Index));
        matrix.resize(rows, cols);
        in.read( (char *) matrix.data() , rows*cols*sizeof(typename Matrix::Scalar) );
        in.close();
    }

    MatrixXd colwise_sig(const MatrixXd& matrix){
      VectorXd matrix_mean = matrix.colwise().mean();
      MatrixXd matrix_std = (matrix - matrix_mean.transpose().replicate(matrix.rows(), 1));
      matrix_std = matrix_std.array().pow(2);
      MatrixXd matrix_sum = matrix_std.colwise().sum();
      matrix_sum *= (1.0/(matrix.rows()-1));
      return matrix_sum.array().sqrt();
    }

    double percentile_v(VectorXd vector, int p) {
      p = p-1;
      if (p < 0) p = 0;
      std::sort(vector.data(),vector.data()+vector.size());
      return vector(std::floor((p/100.0)*vector.size()));
    }

    VectorXd percentile(const MatrixXd& matrix, int p) {
      VectorXd result(matrix.cols());
      for (size_t i = 0; i < matrix.cols(); i++) {
        result(i) = percentile_v(matrix.col(i), p);
      }
      return result;
    }
}
#endif
