#ifndef MEDROPS_MEDROPS_HPP
#define MEDROPS_MEDROPS_HPP

#include <limbo/opt/optimizer.hpp>
#include <fstream>
#include "binary_matrix.hpp"

namespace medrops {

    template <typename Params, typename Model, typename Robot, typename Policy, typename PolicyOptimizer, typename RewardFunction>
    class Medrops {
    public:
        int opt_iters;
        double max_reward;

        Medrops() {}
        ~Medrops() {}

        void execute_and_record_data()
        {
            std::vector<double> R;
            RewardFunction world;
            // Execute best policy so far on robot
            auto obs_new = _robot.execute(_policy, world, Params::medrops::rollout_steps(), R);
            // Append recorded data
            _observations.insert(_observations.end(), obs_new.begin(), obs_new.end());

            // _ofs << R << std::endl;
            for (auto r : R)
                _ofs << r << " ";
            _ofs << std::endl;
        }

        void learn_model()
        {
            _model.learn(_observations);
        }

        void optimize_policy()
        {
            PolicyOptimizer policy_optimizer;

            // For now optimize policy without gradients
            opt_iters = 0;
            max_reward = 0;
            std::cout << "Optimizing policy... " << std::flush;
            Eigen::VectorXd params_star = policy_optimizer(std::bind(&Medrops::_optimize_policy, this, std::placeholders::_1, std::placeholders::_2), _policy.params(), false);
            std::cout << std::endl << "Optimization iterations: " << opt_iters << std::endl;

            // std::cout << "BEST: " << limbo::opt::fun(_optimize_policy(params_star)) << std::endl;

            _policy.set_params(params_star);

            std::cout << "Best parameters: " << params_star.transpose() << std::endl;
            Eigen::write_binary("policy_params.bin", params_star);

            std::vector<double> R;
            RewardFunction world;
            _robot.execute_dummy(_policy, _model, world, Params::medrops::rollout_steps(), R);
            std::cout << "Dummy reward: " << std::accumulate(R.begin(), R.end(),0.0) << std::endl;
            // _ofs << R << " ";
            for (auto r : R)
                _ofs << r << " ";
            _ofs << std::endl;
        }

        void learn(size_t init, size_t iterations)
        {
            _ofs.open("results.dat");
            _policy.set_random_policy();

            std::cout << "Executing random actions..." << std::endl;
            for (size_t i = 0; i < init; i++) {
                execute_and_record_data();
            }

            std::cout << "Starting learning..." << std::endl;
            for (size_t i = 0; i < iterations; i++) {
                std::cout << std::endl << "Learning iteration #" << (i + 1) << std::endl;

                learn_model();
                std::cout << "Learned model..." << std::endl;

                // _model.save_data();
                Eigen::VectorXd errors = get_accuracy();
                std::cout << "Average on errors: " << errors.transpose() << std::endl;

                for (size_t j = 0; j < errors.size(); j++) {
                    if (errors(j) > 1000) {
                        std::cout << "Detected big difference between the approximation and the model, terminating..." << std::endl;
                        exit(-1);
                    }
                }

                optimize_policy();
                std::cout << "Optimized policy..." << std::endl;

                execute_and_record_data();
                std::cout << "Executed action..." << std::endl;
            }
            _ofs.close();

            std::cout << "Averaged errors of the model: " << get_accuracy().transpose() << std::endl;
        }

    protected:
        Robot _robot;
        Policy _policy;
        Model _model;
        std::ofstream _ofs;

        // state, action, prediction
        std::vector<std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd>> _observations;

        limbo::opt::eval_t _optimize_policy(const Eigen::VectorXd& params, bool eval_grad = false)
        {
            RewardFunction world;
            Policy policy;
            policy.set_params(params);

            double r = _robot.predict_policy(policy, _model, world, Params::medrops::rollout_steps());

            opt_iters++;
            if (max_reward < r) max_reward = r;
            if (opt_iters % 10000 == 0) {
              std::cout << opt_iters << "(" << max_reward << ") " << std::flush;
            }

            return limbo::opt::no_grad(r);
        }

        Eigen::VectorXd get_random_vector(size_t dim) const
        {
            Eigen::VectorXd rv = (limbo::tools::random_vector(dim).array()*2-1);
            rv(0) *= 3; rv(1) *= 5; rv(2) *= 6; rv(3) *= M_PI; rv(4) *= 10;
            return rv;
        }

        std::vector<Eigen::VectorXd> random_vectors(size_t dim, size_t q) const
        {
            std::vector<Eigen::VectorXd> result(q);
            for (size_t i = 0; i < q; i++) {
                result[i] = get_random_vector(dim);
            }
            return result;
        }

        Eigen::VectorXd get_accuracy(int evaluations = 100000) const
        {

            std::vector<Eigen::VectorXd> rvs = random_vectors(5, evaluations);

            Eigen::VectorXd errors(4);
            for (int i = 0; i < rvs.size(); i++) {
                Eigen::VectorXd s;
                Eigen::VectorXd m;
                Eigen::VectorXd cm = comp_predict(rvs[i]);
                Eigen::VectorXd gp_query(6);
                gp_query.segment(0,3) = rvs[i].segment(0,3);
                gp_query(3) = std::cos(rvs[i](3));
                gp_query(4) = std::sin(rvs[i](3));
                gp_query(5) = rvs[i](4);
                std::tie(m, s) = _model.predictm(gp_query);

                errors.array() += (m-cm).array().abs();
            }

            return errors.array()/evaluations*1.0;
        }

        Eigen::VectorXd comp_predict(const Eigen::VectorXd& v) const
        {
            double dt = 0.1, t = 0.0;

            boost::numeric::odeint::runge_kutta4<std::vector<double>> ode_stepper;
            std::vector<double> pend_state(4);
            pend_state[0] = v(0);
            pend_state[1] = v(1);
            pend_state[2] = v(2);
            pend_state[3] = v(3);

            boost::numeric::odeint::integrate_const(ode_stepper, std::bind(&Medrops::dynamics, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, v(4)), pend_state, t, dt, dt / 2.0);

            Eigen::VectorXd new_state = Eigen::VectorXd::Map(pend_state.data(), pend_state.size());
            return (new_state.array() - v.segment(0, 4).array());
        }

        void dynamics(const std::vector<double>& x, std::vector<double>& dx, double t, double u) const
        {
            double l = 0.5, m = 0.5, M = 0.5, g = 9.82, b = 0.1;

            dx[0] = x[1];
            dx[1] = (2 * m * l * std::pow(x[2], 2.0) * std::sin(x[3]) + 3 * m * g * std::sin(x[3]) * std::cos(x[3]) + 4 * u - 4 * b * x[1]) / (4 * (M + m) - 3 * m * std::pow(std::cos(x[3]), 2.0));
            dx[2] = (-3 * m * l * std::pow(x[2], 2.0) * std::sin(x[3]) * std::cos(x[3]) - 6 * (M + m) * g * std::sin(x[3]) - 6 * (u - b * x[1]) * std::cos(x[3])) / (4 * l * (m + M) - 3 * m * l * std::pow(std::cos(x[3]), 2.0));
            dx[3] = x[2];
        }
    };
}

#endif
