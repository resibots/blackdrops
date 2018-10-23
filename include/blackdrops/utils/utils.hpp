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
#ifndef UTILS_UTILS_HPP
#define UTILS_UTILS_HPP

#include <string>
#include <sys/stat.h>
#include <utility>

#include <Eigen/Core>

#include <limbo/tools/random_generator.hpp>

namespace blackdrops {
    namespace rng {
        static thread_local limbo::tools::rgen_gauss_t gauss_rng(0., 1.);
        static thread_local limbo::tools::rgen_double_t uniform_rng(0., 1.);
    } // namespace rng

    namespace utils {
        inline Eigen::VectorXd uniform_rand(int size, limbo::tools::rgen_double_t& rgen = rng::uniform_rng)
        {
            return limbo::tools::random_vec(size, rgen);
        }

        inline Eigen::VectorXd gaussian_rand(const Eigen::VectorXd& mean, limbo::tools::rgen_gauss_t& rgen = rng::gauss_rng)
        {
            return limbo::tools::random_vec(mean.size(), rgen) + mean;
        }

        inline Eigen::VectorXd gaussian_rand(const Eigen::VectorXd& mean, const Eigen::MatrixXd& covar, limbo::tools::rgen_gauss_t& rgen = rng::gauss_rng)
        {
            assert(mean.size() == covar.rows() && covar.rows() == covar.cols());

            Eigen::LLT<Eigen::MatrixXd> cholSolver(covar);
            Eigen::MatrixXd transform = cholSolver.matrixL();

            return transform * gaussian_rand(Eigen::VectorXd::Zero(mean.size()), rgen) + mean;
        }

        inline Eigen::VectorXd gaussian_rand(const Eigen::VectorXd& mean, const Eigen::VectorXd& sigma, limbo::tools::rgen_gauss_t& rgen = rng::gauss_rng)
        {
            assert(mean.size() == sigma.size());

            Eigen::MatrixXd covar = Eigen::MatrixXd::Zero(mean.size(), mean.size());
            covar.diagonal() = sigma.array().square();

            return gaussian_rand(mean, covar, rgen);
        }

        inline Eigen::VectorXd gaussian_rand(const Eigen::VectorXd& mean, double sigma, limbo::tools::rgen_gauss_t& rgen = rng::gauss_rng)
        {
            Eigen::VectorXd sig = Eigen::VectorXd::Constant(mean.size(), sigma);

            return gaussian_rand(mean, sig, rgen);
        }

        inline double gaussian_rand(double mean, double sigma, limbo::tools::rgen_gauss_t& rgen = rng::gauss_rng)
        {
            Eigen::VectorXd m(1);
            m << mean;

            return gaussian_rand(m, sigma, rgen)[0];
        }

        inline double angle_dist(double a, double b)
        {
            double theta = b - a;
            while (theta < -M_PI)
                theta += 2 * M_PI;
            while (theta > M_PI)
                theta -= 2 * M_PI;
            return theta;
        }

        // Sample mean and covariance
        inline std::pair<Eigen::VectorXd, Eigen::MatrixXd> sample_statistics(const std::vector<Eigen::VectorXd>& points)
        {
            assert(points.size());

            // Get the sample mean
            Eigen::VectorXd mean = Eigen::VectorXd::Zero(points[0].size());

            for (size_t i = 0; i < points.size(); i++) {
                mean.array() += points[i].array();
            }

            mean = mean.array() / double(points.size());

            // Calculate the sample covariance matrix
            Eigen::MatrixXd cov = Eigen::MatrixXd::Zero(points[0].size(), points[0].size());
            for (size_t i = 0; i < points.size(); i++) {
                cov = cov + points[i] * points[i].transpose();
            }

            cov = (cov.array() - (double(points.size()) * mean * mean.transpose()).array()) / (double(points.size()) - 1.0);

            return {mean, cov};
        }

        inline bool file_exists(const std::string& name)
        {
            struct stat buffer;
            return (stat(name.c_str(), &buffer) == 0);
        }

        bool replace_string(std::string& str, const std::string& from, const std::string& to)
        {
            size_t start_pos = str.find(from);
            if (start_pos == std::string::npos)
                return false;
            str.replace(start_pos, from.length(), to);
            return true;
        }
    } // namespace utils
} // namespace blackdrops

#endif