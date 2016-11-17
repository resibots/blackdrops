#ifndef MEDROPS_KERNEL_LF_OPT
#define MEDROPS_KERNEL_LF_OPT

#include <limbo/model/gp/hp_opt.hpp>
#include <limbo/tools/random_generator.hpp>
#include <cmath>

namespace medrops {

    ///@ingroup model_opt
    ///optimize the likelihood of the kernel only
    template <typename Params, typename Optimizer = limbo::opt::ParallelRepeater<Params, limbo::opt::Rprop<Params>>>
    struct KernelLFOpt : public limbo::model::gp::HPOpt<Params, Optimizer> {
    public:
        template <typename GP>
        void operator()(GP& gp)
        {
            this->_called = true;
            KernelLFOptimization<GP> optimization(gp);
            Optimizer optimizer;
            auto params = optimizer(optimization, gp.kernel_function().h_params(), false);
            gp.kernel_function().set_h_params(params);
            gp.set_lik(limbo::opt::eval(optimization, params));
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


                // TODO: Revisit this, it's not working yet
                if (false) {
                    // sum(((ll - log(curb.std'))./log(curb.ls)).^p);
                    Eigen::VectorXd p = gp.kernel_function().h_params();
                    Eigen::VectorXd ll = p.segment(0, p.size()-1);  // length scales

                    // Std calculation of samples in logspace
                    Eigen::MatrixXd samples = _to_matrix(gp.samples());
                    Eigen::VectorXd samples_mean = samples.colwise().mean();

                    Eigen::MatrixXd samples_std = (samples - samples_mean.transpose().replicate(samples.rows(),1));
                    samples_std = samples_std.array().pow(2);
                    samples_std = samples_std.colwise().sum();
                    samples_std *= (1.0/(samples.rows()-1));

                    samples_std = samples_std.array().sqrt().log();

                    double snr = std::log(500);         // signal to noise threshold
                    double ls = std::log(100);          // length scales threshold
                    double pp = 30;                     // penalty power

                    lik += ((ll - samples_std.transpose()) / ls).array().pow(pp).sum();

                    // f = f + sum(((lsf - lsn)/log(curb.snr)).^p); % signal to noise ratio
                    double lsf = p(p.size()-1);
                    double lsn = std::log(0.01);
                    lik += std::pow((lsf - lsn) / snr, pp);
                }

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
                        Eigen::VectorXd g = gp.kernel_function().grad(gp.samples()[i], gp.samples()[j]);
                        if (i == j)
                            grad += w(i, j) * g * 0.5;
                        else
                            grad += w(i, j) * g;
                    }
                }

                return {lik, grad};
            }

        protected:
            const GP& _original_gp;
        };
    };
}

#endif
