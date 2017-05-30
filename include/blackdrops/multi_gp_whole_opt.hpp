//| Copyright Inria May 2015
//| This project has received funding from the European Research Council (ERC) under
//| the European Union's Horizon 2020 research and innovation programme (grant
//| agreement No 637972) - see http://www.resibots.eu
//|
//| Contributor(s):
//|   - Jean-Baptiste Mouret (jean-baptiste.mouret@inria.fr)
//|   - Antoine Cully (antoinecully@gmail.com)
//|   - Kontantinos Chatzilygeroudis (konstantinos.chatzilygeroudis@inria.fr)
//|   - Federico Allocati (fede.allocati@gmail.com)
//|   - Vaios Papaspyros (b.papaspyros@gmail.com)
//|   - Roberto Rama (bertoski@gmail.com)
//|
//| This software is a computer library whose purpose is to optimize continuous,
//| black-box functions. It mainly implements Gaussian processes and Bayesian
//| optimization.
//| Main repository: http://github.com/resibots/limbo
//| Documentation: http://www.resibots.eu/limbo
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
#ifndef MULTI_GP_WHOLE_OPT_HPP
#define MULTI_GP_WHOLE_OPT_HPP

#include <limbo/model/gp/hp_opt.hpp>
#include <limbo/tools/random_generator.hpp>
#include <blackdrops/binary_matrix.hpp>
#include <limbo/model/gp/kernel_lf_opt.hpp>
#include <blackdrops/kernel_lf_opt.hpp>
#include <limits>

namespace blackdrops {

    template <typename Params, typename Optimizer = limbo::opt::Rprop<Params>, typename OptimizerLocal = limbo::model::gp::KernelLFOpt<Params, limbo::opt::Rprop<Params>>>
    struct MultiGPWholeLFOpt : public limbo::model::gp::HPOpt<Params, Optimizer> {
    public:
        template <typename GP>
        void operator()(GP& gp)
        {
            this->_called = true;
            MultiGPWholeLFOptimization<GP> optimization(gp);
            Optimizer optimizer;
            // std::cout << "kernel: " << gp.kernel_h_params().transpose().array().exp() << std::endl;
            Eigen::VectorXd params = optimizer(optimization, gp.mean_function().h_params(), true);
            gp.mean_function().set_h_params(params);
            // std::cout << "mean: " << gp.mean_h_params().array().exp().transpose() << std::endl;
            std::cout << "mean: " << gp.mean_function().h_params().transpose() << std::endl;
            gp.recompute(true, false);
            // std::cout << "kernel after: " << gp.kernel_h_params().transpose().array().exp() << std::endl;

            auto& gps = gp.gp_models();
            // for (auto& small_gp : gps)
            tbb::parallel_for(size_t(0), gps.size(), size_t(1), [&](size_t i) {
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
                tbb::parallel_for(size_t(0), small_gps.size(), size_t(1), [&](size_t i) {
                    OptimizerLocal hp_optimize;
                    hp_optimize(small_gps[i]);
                });

                long double lik_all = 0.0;
                Eigen::VectorXd grad_all;

                auto gps = gp_all.gp_models();

                for (auto gp : gps) {
                    // size_t n = gp.obs_mean().rows();
                    //
                    // // --- cholesky ---
                    // // see:
                    // // http://xcorr.net/2008/06/11/log-determinant-of-positive-definite-matrices-in-matlab/
                    // Eigen::MatrixXd l = gp.matrixL();
                    // long double det = 2 * l.diagonal().array().log().sum();
                    //
                    // double a = (gp.obs_mean().transpose() * gp.alpha())
                    //                .trace(); // generalization for multi dimensional observation
                    // // std::cout << " a: " << a << " det: " << det << std::endl;
                    // long double lik = -0.5 * a - 0.5 * det - 0.5 * n * log(2 * M_PI);

                    long double lik = gp.compute_lik();

                    lik_all += std::exp(-lik);
                    if (!compute_grad)
                        continue;
                }

                // std::cout << -std::log(lik_all) << " vs " << (initial_params - params).squaredNorm() << std::endl;
                lik_all = -std::log(lik_all); // - (initial_params - params).norm();
                // std::cout << params.transpose().array().exp() << " ---> " << lik_all << std::endl;
                // std::cin.get();

                if (!compute_grad)
                    return limbo::opt::no_grad(lik_all);

                return {lik_all, grad_all};
            }

        protected:
            const GP& _original_gp;
        };
    };
}

#endif
