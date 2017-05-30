#include <limbo/limbo.hpp>

#include <blackdrops/multi_gp.hpp>
#include <blackdrops/parallel_gp.hpp>
#include <blackdrops/multi_gp_whole_opt.hpp>
#include <blackdrops/cmaes.hpp>

#include <boost/numeric/odeint.hpp>

struct Params {
    struct kernel : public limbo::defaults::kernel {
        BO_PARAM(double, noise, 0.01);
        BO_PARAM(bool, optimize_noise, true);
    };

    struct kernel_squared_exp_ard : public limbo::defaults::kernel_squared_exp_ard {
    };

    struct opt_rprop : public limbo::defaults::opt_rprop {
        BO_PARAM(int, iterations, 150);
    };

    struct opt_parallelrepeater : public limbo::defaults::opt_parallelrepeater {
        BO_PARAM(int, repeats, 2);
    };

    struct opt_cmaes : public limbo::defaults::opt_cmaes {
        BO_PARAM(double, max_fun_evals, 100);
        // BO_PARAM(double, fun_tolerance, 0.1);
        BO_PARAM(int, restarts, 5);
        BO_PARAM(int, elitism, 1);
        BO_PARAM(int, variant, aBIPOP_CMAES);
        BO_PARAM(bool, fun_compute_initial, true);
    };

    struct opt_nloptnograd {
        BO_PARAM(int, iterations, 100);
    };
};

struct MeanModel {
    typedef std::vector<double> ode_state_type;

    Eigen::VectorXd operator()(const Eigen::VectorXd& v) const
    {
        double dt = 0.1;
        boost::numeric::odeint::runge_kutta4<ode_state_type> ode_stepper;

        ode_state_type cp_state(4, 0.0);
        cp_state[0] = v(0);
        cp_state[1] = v(1);
        cp_state[2] = v(2);
        cp_state[3] = std::atan2(v(4), v(3));
        double u = v(5);

        Eigen::VectorXd init = Eigen::VectorXd::Map(cp_state.data(), cp_state.size());

        boost::numeric::odeint::integrate_const(ode_stepper,
            std::bind(&MeanModel::dynamics, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, u),
            cp_state, 0.0, dt, dt / 2.0);

        Eigen::VectorXd final = Eigen::VectorXd::Map(cp_state.data(), cp_state.size());
        return (final - init);
    }

    /* The rhs of x' = f(x) */
    void dynamics(const ode_state_type& x, ode_state_type& dx, double t, double u) const
    {
        double l = 0.5, m = 0.5, M = 0.5, g = 9.82, b = 0.1;

        dx[0] = x[1];
        dx[1] = (2 * m * l * std::pow(x[2], 2.0) * std::sin(x[3]) + 3 * m * g * std::sin(x[3]) * std::cos(x[3]) + 4 * u - 4 * b * x[1]) / (4 * (M + m) - 3 * m * std::pow(std::cos(x[3]), 2.0));
        dx[2] = (-3 * m * l * std::pow(x[2], 2.0) * std::sin(x[3]) * std::cos(x[3]) - 6 * (M + m) * g * std::sin(x[3]) - 6 * (u - b * x[1]) * std::cos(x[3])) / (4 * l * (m + M) - 3 * m * l * std::pow(std::cos(x[3]), 2.0));
        dx[3] = x[2];
    }

    std::vector<Eigen::VectorXd> bounds() const
    {
        std::vector<Eigen::VectorXd> b;
        Eigen::VectorXd tmp(2);
        tmp << -6, 6;
        b.push_back(tmp);
        tmp << -2, 2;
        b.push_back(tmp);
        tmp << -6, 6;
        b.push_back(tmp);
        tmp << -1, 1;
        b.push_back(tmp);
        tmp << -1, 1;
        b.push_back(tmp);
        tmp << -10, 10;
        b.push_back(tmp);
        return b;
    }

    int dims() const
    {
        return 6;
    }
};

struct MeanFunc {
    typedef std::vector<double> ode_state_type;

    MeanFunc(int dim_out = 1)
    {
        _params = Eigen::VectorXd::Zero(3);
        _params(0) = 0.45; //l
        _params(1) = 0.55; //m
        _params(2) = 0.4; //M
    }

    template <typename GP>
    Eigen::VectorXd operator()(const Eigen::VectorXd& v, const GP& gp) const
    {
        double dt = 0.1;
        boost::numeric::odeint::runge_kutta4<ode_state_type> ode_stepper;

        ode_state_type cp_state(4, 0.0);
        cp_state[0] = v(0);
        cp_state[1] = v(1);
        cp_state[2] = v(2);
        cp_state[3] = std::atan2(v(4), v(3));
        double u = v(5);

        Eigen::VectorXd init = Eigen::VectorXd::Map(cp_state.data(), cp_state.size());

        boost::numeric::odeint::integrate_const(ode_stepper,
            std::bind(&MeanFunc::dynamics, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, u),
            cp_state, 0.0, dt, dt / 2.0);

        Eigen::VectorXd final = Eigen::VectorXd::Map(cp_state.data(), cp_state.size());
        return (final - init);
    }

    Eigen::VectorXd h_params() const { return _params; }

    void set_h_params(const Eigen::VectorXd& params)
    {
        _params = params;
    }

    /* The rhs of x' = f(x) */
    void dynamics(const ode_state_type& x, ode_state_type& dx, double t, double u) const
    {
        double l = _params(0), m = _params(1), M = _params(2), g = 9.82, b = 0.1;

        dx[0] = x[1];
        dx[1] = (2 * m * l * std::pow(x[2], 2.0) * std::sin(x[3]) + 3 * m * g * std::sin(x[3]) * std::cos(x[3]) + 4 * u - 4 * b * x[1]) / (4 * (M + m) - 3 * m * std::pow(std::cos(x[3]), 2.0));
        dx[2] = (-3 * m * l * std::pow(x[2], 2.0) * std::sin(x[3]) * std::cos(x[3]) - 6 * (M + m) * g * std::sin(x[3]) - 6 * (u - b * x[1]) * std::cos(x[3])) / (4 * l * (m + M) - 3 * m * l * std::pow(std::cos(x[3]), 2.0));
        dx[3] = x[2];
    }

protected:
    Eigen::VectorXd _params;
};

template <typename Function>
void benchmark(const std::string& name)
{
    Function func;
    std::vector<int> dims;
    if (func.dims() > 0)
        dims.push_back(func.dims());
    else {
        for (int i = 1; i <= 2; i *= 2) {
            dims.push_back(i);
        }
    }

    int N_test = 10000;

    for (size_t dim = 0; dim < dims.size(); dim++) {
        std::vector<Eigen::VectorXd> bounds = func.bounds();
        bool one_bound = (bounds.size() == 1);
        int D = dims[dim];

        blackdrops::MultiGP<Params, limbo::model::GP, limbo::kernel::SquaredExpARD<Params>, MeanFunc, blackdrops::MultiGPWholeLFOpt<Params, limbo::opt::NLOptNoGrad<Params, nlopt::LN_SBPLX>>> gp;
        blackdrops::ParallelGP<Params, limbo::model::GP, limbo::kernel::SquaredExpARD<Params>, MeanFunc, limbo::model::gp::KernelLFOpt<Params>> gp_old;

        for (int N = 40; N <= 160; N += 40) {
            std::cout << name << " in dim: " << D << " and # of points: " << N << std::endl;
            std::string file_name = name + "_" + std::to_string(D) + "_" + std::to_string(N);

            std::vector<Eigen::VectorXd> points, obs;

            // for (int i = 0; i < N; i++) {
            //     Eigen::VectorXd p = limbo::tools::random_vector(D); //.array() * 10.24 - 5.12;
            //     if (one_bound)
            //         p = p.array() * (bounds[0](1) - bounds[0](0)) + bounds[0](0);
            //     else {
            //         for (int j = 0; j < D; j++) {
            //             p(j) = p(j) * (bounds[j](1) - bounds[j](0)) + bounds[j](0);
            //         }
            //     }
            //
            //     points.push_back(p);
            //     Eigen::VectorXd ob = func(p);
            //     // TO-DO: Put noise according to observations width
            //     // ob << func(p); // + gaussian_rand(0.0, 1.0);
            //     // std::cout << ob << std::endl;
            //     obs.push_back(ob);
            // }

            Eigen::MatrixXd data_comp;
            Eigen::read_binary("blackdrops_data.bin", data_comp);

            size_t limit = N;
            std::cout << "Loading " << limit << "/" << data_comp.rows() << " rows from file." << std::endl;

            // std::vector<Eigen::VectorXd> samples_comp(limit), observations_comp(limit);
            for (size_t i = 0; i < limit; i++) {
                points.push_back(data_comp.row(i).segment(0, 6));
                obs.push_back(data_comp.row(i).segment(6, 4));
                // std::cout << obs.back().transpose() << " vs " << func(points.back()).transpose() << std::endl;
            }

            // Eigen::MatrixXd cov = spt::sample_covariance(obs);
            // double sigma = std::sqrt(cov(0, 0)) / 20.0;
            //
            // std::cout << "Adding noise of: " << sigma << std::endl;
            //
            // for (int i = 0; i < N; i++)
            //     obs[i] = obs[i].array() + gaussian_rand(0.0, sigma);

            // blackdrops::MultiGP<Params, limbo::kernel::SquaredExpARD<Params>, MeanFunc, blackdrops::MultiGPWholeLFOpt<Params, limbo::opt::NLOptNoGrad<Params, nlopt::LN_SBPLX>>> gp;

            auto start = std::chrono::high_resolution_clock::now();
            gp.compute(points, obs, true);
            gp.optimize_hyperparams();
            // Eigen::VectorXd mean_pp(2);
            // mean_pp << std::log(0.5), std::log(0.5);
            // gp.set_mean_h_params(mean_pp);
            // gp.recompute(true, false);
            // auto& gps = gp.gp_models();
            // // for (auto& small_gp : gps)
            // tbb::parallel_for(size_t(0), gps.size(), size_t(1), [&](size_t i) {
            //     limbo::model::gp::KernelLFOpt<Params, limbo::opt::Rprop<Params>> hp_optimize;
            //     hp_optimize(gps[i]);
            // });
            // std::cout << "mean: " << gp.mean_h_params().array().exp().transpose() << std::endl;

            // Eigen::VectorXd ppk = gp.h_params();
            // std::cout << "gp : ";
            // for (int j = 0; j < ppk.size() - 2; j++)
            //     std::cout << std::exp(ppk(j)) << " ";
            // std::cout << std::exp(2 * ppk(ppk.size() - 2)) << " " << std::exp(2 * ppk(ppk.size() - 1)) << std::endl;
            auto time1 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count();
            std::cout << "Time in secs: " << time1 / double(1000.0) << std::endl;

            // blackdrops::MultiGP<Params, limbo::kernel::SquaredExpARD<Params>, MeanFunc, blackdrops::MultiGPWholeLFOpt<Params, limbo::opt::CustomCmaes<Params>>> gp_old;

            // blackdrops::ParallelGP<Params, limbo::kernel::SquaredExpARD<Params>, MeanFunc, limbo::model::gp::KernelLFOpt<Params, limbo::opt::CustomCmaes<SecondParams>>> gp_old;
            Eigen::VectorXd mean_pp(3);
            mean_pp << std::log(0.5), std::log(0.5), std::log(0.5);
            gp_old.mean_function().set_h_params(mean_pp);
            // blackdrops::MultiGP<Params, limbo::kernel::SquaredExpARD<Params>, MeanFunc, blackdrops::MultiGPMeanLFOpt<Params, limbo::opt::CustomCmaes<Params>>> gp_old;
            start = std::chrono::high_resolution_clock::now();
            gp_old.compute(points, obs, false);
            gp_old.optimize_hyperparams();
            // Eigen::VectorXd pk = gp_old.kernel_function().h_params();
            // std::cout << "old_gp : ";
            // for (int j = 0; j < pk.size() - 2; j++)
            //     std::cout << std::exp(pk(j)) << " ";
            // std::cout << std::exp(2 * pk(pk.size() - 2)) << " " << std::exp(2 * pk(pk.size() - 1)) << std::endl;
            time1 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count();
            std::cout << "Time (old gp) in secs: " << time1 / double(1000.0) << std::endl;

            std::vector<Eigen::VectorXd> test_points, test_obs, stgp, old_gp;
            std::vector<double> stgp_err, old_gp_err;
            for (int i = 0; i < N_test; i++) {
                Eigen::VectorXd p = limbo::tools::random_vector(D);
                if (one_bound)
                    p = p.array() * (bounds[0](1) - bounds[0](0)) + bounds[0](0);
                else {
                    for (int j = 0; j < D; j++) {
                        p(j) = p(j) * (bounds[j](1) - bounds[j](0)) + bounds[j](0);
                    }
                }

                test_points.push_back(p);

                Eigen::VectorXd ob = func(p); //(1);
                // ob << func(p);
                test_obs.push_back(ob);
            }

            start = std::chrono::high_resolution_clock::now();
            double err = 0.0;
            Eigen::VectorXd avg_sigma = Eigen::VectorXd::Zero(test_obs[0].size());
            for (int i = 0; i < N_test; i++) {
                Eigen::VectorXd mm;
                Eigen::VectorXd ss;
                std::tie(mm, ss) = gp.query(test_points[i]);
                stgp.push_back(mm);
                stgp_err.push_back((stgp.back() - test_obs[i]).norm());
                err += stgp_err.back();
                avg_sigma.array() += ss.array();
            }
            time1 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count();
            std::cout << "Time (query) in ms: " << time1 / double(N_test) << " MSE: " << err / double(N_test) << std::endl;
            std::cout << "Average sigma: " << avg_sigma.transpose().array() / double(N_test) << std::endl;

            start = std::chrono::high_resolution_clock::now();
            err = 0.0;
            avg_sigma = Eigen::VectorXd::Zero(test_obs[0].size());
            for (int i = 0; i < N_test; i++) {
                Eigen::VectorXd mm;
                Eigen::VectorXd ss;
                std::tie(mm, ss) = gp_old.query(test_points[i]);
                old_gp.push_back(mm);
                old_gp_err.push_back((old_gp.back() - test_obs[i]).norm());
                err += old_gp_err.back();
                avg_sigma.array() += ss.array();
            }
            time1 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count();
            std::cout << "Time (old_gp-query) in ms: " << time1 / double(N_test) << " MSE: " << err / double(N_test) << std::endl;
            std::cout << "Average sigma: " << avg_sigma.transpose().array() / double(N_test) << std::endl;

            std::cout << "Saving data..." << std::endl;

            std::ofstream ofs(file_name + "_gp.dat");
            std::ofstream ofs_old(file_name + "_gp_old.dat");
            std::ofstream ofs_real(file_name + "_real.dat");
            int pp = 4000;
            for (int i = 0; i < pp; ++i) {
                Eigen::VectorXd v = limbo::tools::random_vector(D);
                if (one_bound)
                    v = v.array() * (bounds[0](1) - bounds[0](0)) + bounds[0](0);
                else {
                    for (int j = 0; j < D; j++) {
                        v(j) = v(j) * (bounds[j](1) - bounds[j](0)) + bounds[j](0);
                    }
                }
                Eigen::VectorXd mu;
                Eigen::VectorXd sigma;
                std::tie(mu, sigma) = gp.query(v);
                // an alternative (slower) is to query mu and sigma separately:
                //  double mu = gp.mu(v)[0]; // mu() returns a 1-D vector
                //  double s2 = gp.sigma(v);
                ofs << v.transpose() << " " << mu.transpose() << " " << std::sqrt(sigma[0]) << std::endl;

                std::tie(mu, sigma) = gp_old.query(v);
                ofs_old << v.transpose() << " " << mu.transpose() << " " << std::sqrt(sigma[0]) << std::endl;

                Eigen::VectorXd val = func(v);
                ofs_real << v.transpose() << " " << val.transpose() << " 0" << std::endl;
            }

            std::ofstream ofs_data(file_name + "_data.dat");
            for (size_t i = 0; i < points.size(); ++i)
                ofs_data << points[i].transpose() << " " << obs[i].transpose() << std::endl;

            std::cout << "Data saved...!" << std::endl;
        }
    }
}

int main()
{
    limbo::tools::par::init();
    // static tbb::task_scheduler_init init(1);
    benchmark<MeanModel>("mean_model");
    return 0;
}
