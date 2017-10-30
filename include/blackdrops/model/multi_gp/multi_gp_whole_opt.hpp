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
#ifndef ONLYMI
                    gp.recompute(true, false);
                    auto& gps = gp.gp_models();
                    // for (auto& small_gp : gps)
                    tbb::parallel_for(size_t(0), gps.size(), size_t(1), [&](size_t i) {
                        OptimizerLocal hp_optimize;
                        hp_optimize(gps[i]);
                    });
#else
                    gp.recompute(true, true);
#endif
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

#ifndef ONLYMI
                        gp_all.recompute(true, false);

                        auto& small_gps = gp_all.gp_models();
                        tbb::parallel_for(size_t(0), small_gps.size(), size_t(1), [&](size_t i) {
                            OptimizerLocal hp_optimize;
                            hp_optimize(small_gps[i]);
                        });
#else
                        gp_all.recompute(true, true);
#endif

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
        }
    }
}
#endif
