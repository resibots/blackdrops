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
#ifndef BLACKDROPS_MODEL_GP_KERNEL_LF_OPT
#define BLACKDROPS_MODEL_GP_KERNEL_LF_OPT

#include <cmath>
#include <limbo/model/gp/hp_opt.hpp>
#include <limbo/tools/random_generator.hpp>

namespace blackdrops {
    namespace model {
        namespace gp {
            ///optimize the likelihood of the kernel only
            template <typename Params, typename Optimizer = limbo::opt::Rprop<Params>>
            struct KernelLFOpt : public limbo::model::gp::HPOpt<Params, Optimizer> {
            public:
                template <typename GP>
                void operator()(GP& gp)
                {
                    this->_called = true;
                    KernelLFOptimization<GP> optimization(gp);
                    Optimizer optimizer;
                    Eigen::VectorXd params = optimizer(optimization, gp.kernel_function().h_params(), false);
                    gp.kernel_function().set_h_params(params);
                    gp.set_log_lik(limbo::opt::eval(optimization, params));
                    gp.recompute(false);
                }

            protected:
                template <typename GP>
                struct KernelLFOptimization {
                public:
                    KernelLFOptimization(const GP& gp) : _original_gp(gp) {}

                    Eigen::MatrixXd _to_matrix(const std::vector<Eigen::VectorXd>& xs) const
                    {
                        Eigen::MatrixXd result(xs.size(), xs[0].size());
                        for (size_t i = 0; i < (size_t)result.rows(); ++i) {
                            result.row(i) = xs[i];
                        }
                        return result;
                    }
                    Eigen::MatrixXd _to_matrix(std::vector<Eigen::VectorXd>& xs) const { return _to_matrix(xs); }

                    limbo::opt::eval_t operator()(const Eigen::VectorXd& params, bool compute_grad) const
                    {
                        GP gp(this->_original_gp);
                        gp.kernel_function().set_h_params(params);

                        gp.recompute(false);

                        size_t n = gp.obs_mean().rows();

                        // --- cholesky ---
                        // see:
                        // http://xcorr.net/2008/06/11/log-determinant-of-positive-definite-matrices-in-matlab/
                        Eigen::MatrixXd l = gp.matrixL();
                        long double det = 2 * l.diagonal().array().log().sum();

                        double a = (gp.obs_mean().transpose() * gp.alpha())
                                       .trace(); // generalization for multi dimensional observation
                        // std::cout<<" a: "<<a <<" det: "<< det<<std::endl;
                        double lik = -0.5 * a - 0.5 * det - 0.5 * n * log(2 * M_PI);

                        // sum(((ll - log(curb.std'))./log(curb.ls)).^p);
                        Eigen::VectorXd p = gp.kernel_function().h_params();
                        Eigen::VectorXd ll = p.segment(0, p.size() - 2); // length scales

                        // Std calculation of samples in logspace
                        Eigen::MatrixXd samples = _to_matrix(gp.samples());
                        Eigen::MatrixXd samples_std = Eigen::colwise_sig(samples).array().log();

                        double snr = std::log(500); // signal to noise threshold
                        double ls = std::log(100); // length scales threshold
                        size_t pp = 30; // penalty power

                        lik -= ((ll - samples_std.transpose()) / ls).array().pow(pp).sum();

                        // f = f + sum(((lsf - lsn)/log(curb.snr)).^p); % signal to noise ratio
                        double lsf = p(p.size() - 2);
                        double lsn = p(p.size() - 1); //std::log(0.01);
                        lik -= std::pow((lsf - lsn) / snr, pp);

                        if (!compute_grad)
                            return limbo::opt::no_grad(lik);

                        // K^{-1} using Cholesky decomposition
                        Eigen::MatrixXd w = Eigen::MatrixXd::Identity(n, n);

                        gp.matrixL().template triangularView<Eigen::Lower>().solveInPlace(w);
                        gp.matrixL().template triangularView<Eigen::Lower>().transpose().solveInPlace(w);

                        // alpha * alpha.transpose() - K^{-1}
                        w = gp.alpha() * gp.alpha().transpose() - w;

                        // only compute half of the matrix (symmetrical matrix)
                        Eigen::VectorXd grad = Eigen::VectorXd::Zero(params.size());
                        for (size_t i = 0; i < n; ++i) {
                            for (size_t j = 0; j <= i; ++j) {
                                Eigen::VectorXd g = gp.kernel_function().grad(gp.samples()[i], gp.samples()[j], i, j);
                                if (i == j)
                                    grad += w(i, j) * g * 0.5;
                                else
                                    grad += w(i, j) * g;
                            }
                        }

                        // Gradient update with penalties
                        /// df(li) += (p * ((ll - log(curb.std')).^(p-1))) / (log(curb.ls)^p);
                        Eigen::VectorXd grad_ll = pp * (ll - samples_std.transpose()).array().pow(pp - 1) / std::pow(ls, pp);
                        grad.segment(0, grad.size() - 2) = grad.segment(0, grad.size() - 2) - grad_ll;

                        /// df(sfi) = df(sfi) + p*(lsf - lsn).^(p-1)/log(curb.snr)^p;
                        double mgrad_v = pp * std::pow((lsf - lsn), pp - 1) / std::pow(snr, pp);
                        grad(grad.size() - 2) = grad(grad.size() - 2) - mgrad_v;

                        // NOTE: This is for the noise calculation
                        // df(end) = df(end) - p * sum((lsf - lsn).^ (p - 1) / log(curb.snr) ^ p);
                        grad(grad.size() - 1) += pp * std::pow((lsf - lsn), pp - 1) / std::pow(snr, pp);

                        return {lik, grad};
                    }

                protected:
                    const GP& _original_gp;
                };
            };
        } // namespace gp
    } // namespace model
} // namespace blackdrops
#endif
