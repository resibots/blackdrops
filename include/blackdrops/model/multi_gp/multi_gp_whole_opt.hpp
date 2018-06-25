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
#ifndef BLACKDROPS_MODEL_MULTI_GP_MULTI_GP_WHOLE_OPT_HPP
#define BLACKDROPS_MODEL_MULTI_GP_MULTI_GP_WHOLE_OPT_HPP

#include <Eigen/binary_matrix.hpp>
#include <blackdrops/model/gp/kernel_lf_opt.hpp>
#include <limbo/model/gp/hp_opt.hpp>
#include <limbo/model/gp/kernel_lf_opt.hpp>
#include <limbo/tools/random_generator.hpp>
#include <limits>

namespace blackdrops {
    namespace model {
        namespace multi_gp {
            template <typename Params, typename Optimizer = limbo::opt::Rprop<Params>, typename OptimizerLocal = limbo::model::gp::KernelLFOpt<Params, limbo::opt::Rprop<Params>>>
            struct MultiGPWholeLFOpt : public limbo::model::gp::HPOpt<Params, Optimizer> {
            public:
                template <typename GP>
                void operator()(GP& gp)
                {
                    this->_called = true;
                    MultiGPWholeLFOptimization<GP> optimization(gp);
                    Optimizer optimizer;
                    Eigen::VectorXd params = optimizer(optimization, gp.mean_function().h_params(), true);
                    gp.mean_function().set_h_params(params);
                    // std::cout << "mean: " << gp.mean_h_params().array().exp().transpose() << std::endl;
                    std::cout << "mean: " << gp.mean_function().h_params().transpose() << std::endl;
                    gp.recompute(true, false);
                    auto& gps = gp.gp_models();
                    // for (auto& small_gp : gps)
                    limbo::tools::par::loop(0, gps.size(), [&](size_t i) {
                        OptimizerLocal hp_optimize;
                        hp_optimize(gps[i]);
                    });
                    std::cout << "Likelihood: " << limbo::opt::eval(optimization, params) << std::endl;
                }

            protected:
                template <typename GP>
                struct MultiGPWholeLFOptimization {
                public:
                    MultiGPWholeLFOptimization(const GP& gp) : _original_gp(gp) {}

                    limbo::opt::eval_t operator()(const Eigen::VectorXd& params, bool compute_grad) const
                    {
                        GP gp_all(this->_original_gp);
                        // Eigen::VectorXd initial_params = gp_all.mean_h_params();
                        gp_all.mean_function().set_h_params(params);

                        gp_all.recompute(true, false);

                        auto& small_gps = gp_all.gp_models();
                        limbo::tools::par::loop(0, small_gps.size(), [&](size_t i) {
                            OptimizerLocal hp_optimize;
                            hp_optimize(small_gps[i]);
                        });

                        long double lik_all = 0.0;
                        Eigen::VectorXd grad_all;

                        auto gps = gp_all.gp_models();

                        for (auto gp : gps) {
                            long double lik = gp.compute_log_lik();

                            // lik_all += std::exp(-lik);
                            lik_all += std::exp(lik); //(1. / lik);
                            if (!compute_grad)
                                continue;
                        }

                        // std::cout << -std::log(lik_all) << " vs " << (initial_params - params).squaredNorm() << std::endl;
                        // lik_all = -std::log(lik_all); // - (initial_params - params).norm();
                        // lik_all = 1. / lik_all;
                        lik_all = std::log(lik_all);

                        if (!compute_grad)
                            return limbo::opt::no_grad(lik_all);

                        return {lik_all, grad_all};
                    }

                protected:
                    const GP& _original_gp;
                };
            };
        } // namespace multi_gp
    } // namespace model
} // namespace blackdrops
#endif
