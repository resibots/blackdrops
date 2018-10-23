//| Copyright Inria July 2017
//| This project has received funding from the European Research Council (ERC) under
//| the European Union's Horizon 2020 research and innovation programme (grant
//| agreement No 637972) - see http://www.resibots.eu
//|
//| Contributor(s):
//|   - Konstantinos Chatzilygeroudis (konstantinos.chatzilygeroudis@inria.fr)
//|   - Rituraj Kaushik (rituraj.kaushik@inria.fr)
//|   - Roberto Rama (bertoski@gmail.com)
//|
//| This software is the implementation of the Black-DROPS algorithm, which is
//| a model-based policy search algorithm with the following main properties:
//|   - uses Gaussian processes (GPs) to model the dynamics of the robot/system
//|   - takes into account the uncertainty of the dynamical model when
//|                                                      searching for a policy
//|   - is data-efficient or sample-efficient; i.e., it requires very small
//|     interaction time with the system to find a working policy (e.g.,
//|     around 16-20 seconds to learn a policy for the cart-pole swing up task)
//|   - when several cores are available, it can be faster than analytical
//|                                                    approaches (e.g., PILCO)
//|   - it imposes no constraints on the type of the reward function (it can
//|                                                  also be learned from data)
//|   - it imposes no constraints on the type of the policy representation
//|     (any parameterized policy can be used --- e.g., dynamic movement
//|                                              primitives or neural networks)
//|
//| Main repository: http://github.com/resibots/blackdrops
//| Preprint: https://arxiv.org/abs/1703.07261
//|
//| This software is governed by the CeCILL-C license under French law and
//| abiding by the rules of distribution of free software.  You can  use,
//| modify and/ or redistribute the software under the terms of the CeCILL-C
//| license as circulated by CEA, CNRS and INRIA at the following URL
//| "http://www.cecill.info".
//|
//| As a counterpart to the access to the source code and  rights to copy,
//| modify and redistribute granted by the license, users are provided only
//| with a limited warranty  and the software's author,  the holder of the
//| economic rights,  and the successive licensors  have only  limited
//| liability.
//|
//| In this respect, the user's attention is drawn to the risks associated
//| with loading,  using,  modifying and/or developing or reproducing the
//| software by the user in light of its specific status of free software,
//| that may mean  that it is complicated to manipulate,  and  that  also
//| therefore means  that it is reserved for developers  and  experienced
//| professionals having in-depth computer knowledge. Users are therefore
//| encouraged to load and test the software's suitability as regards their
//| requirements in conditions enabling the security of their systems and/or
//| data to be ensured and,  more generally, to use and operate it in the
//| same conditions as regards security.
//|
//| The fact that you are presently reading this means that you have had
//| knowledge of the CeCILL-C license and that you accept its terms.
//|
#ifndef EIGEN_BINARY_MATRIX
#define EIGEN_BINARY_MATRIX

#include <cmath>
#include <fstream>

#include <Eigen/Core>

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
