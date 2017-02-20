#ifndef MEDROPS_MEDROPS_HPP
#define MEDROPS_MEDROPS_HPP

#include "binary_matrix.hpp"
#include <chrono>
#include <fstream>
#include <limits>
#include <limbo/opt/optimizer.hpp>

namespace medrops {

    template <typename Params, typename Model, typename Robot, typename Policy, typename PolicyOptimizer, typename RewardFunction>
    class Medrops {
    public:
        int _opt_iters;
        double _max_reward;
        double _max_simu_reward;
        double _max_real_reward;
        Eigen::VectorXd _max_params;
        double _boundary;
        Eigen::VectorXd old_params;
        Eigen::VectorXd old_starting;

        Medrops() : _best(-std::numeric_limits<double>::max()) {}
        ~Medrops() {}

        void execute_and_record_data()
        {
            std::vector<double> R;
            RewardFunction world;
            // Execute best policy so far on robot
            auto obs_new = _robot.execute(_policy, world, Params::medrops::rollout_steps(), R);

            // Check if it is better than the previous best
            double r_new = std::accumulate(R.begin(), R.end(), 0.0);
            if (r_new > _best) {
                _best = r_new;
                _params_starting = _policy.params();
            }

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

        void optimize_policy(size_t i)
        {
            PolicyOptimizer policy_optimizer;

            // For now optimize policy without gradients
            Eigen::VectorXd params_star;
            Eigen::VectorXd params_starting = _policy.params(); //_params_starting;
            if (_random_policies)
                params_starting = _params_starting;
            Eigen::write_binary("policy_params_starting_" + std::to_string(i) + ".bin", params_starting);

            _opt_iters = 0;
            _max_reward = 0;
            _max_simu_reward = 0;
            _max_real_reward = 0;
            if (_boundary == 0) {
                std::cout << "Optimizing policy... " << std::flush;
                params_star = policy_optimizer(
                    std::bind(&Medrops::_optimize_policy, this, std::placeholders::_1, std::placeholders::_2),
                    params_starting,
                    false);
            }
            else {
                std::cout << "Optimizing policy bounded to [-" << _boundary << ", " << _boundary << "]... " << std::flush;
                params_star = policy_optimizer(
                    std::bind(&Medrops::_optimize_policy, this, std::placeholders::_1, std::placeholders::_2),
                    params_starting,
                    true);
            }
            if (Params::verbose())
                std::cout << _opt_iters << "(" << _max_reward << ")" << std::endl; //", " << _max_simu_reward << ", " << _max_real_reward << ") " << std::endl;
            else
                std::cout << std::endl;
            // std::cout << _opt_iters << "(" << _max_reward << ") " << std::endl;
            std::cout << "Optimization iterations: " << _opt_iters << std::endl;
            // std::cout << "Max parameters: " << _max_params.transpose() << std::endl;

            // Since we are optimizing a noisy function, it is not good to keep the best ever found
            // if (Params::opt_cmaes::elitism() == 0)
            //     params_star = _max_params;

            _policy.normalize(_model);
            _policy.set_params(params_star);

            // std::cout << "Best parameters: " << _policy.params().transpose() << std::endl;
            Eigen::write_binary("policy_params_" + std::to_string(i) + ".bin", _policy.params());

            std::vector<double> R;
            RewardFunction world;
            _robot.execute_dummy(_policy, _model, world, Params::medrops::rollout_steps(), R);
            std::cout << "Dummy reward: " << std::accumulate(R.begin(), R.end(), 0.0) << std::endl;

            for (auto r : R)
                _ofs << r << " ";
            _ofs << std::endl;
        }

        void learn(size_t init, size_t iterations, bool random_policies = false)
        {
            _boundary = Params::medrops::boundary();
            _random_policies = random_policies;
            _ofs.open("results.dat");
            _ofs_opt.open("times.dat");
            _ofs_model.open("times_model.dat");
            _policy.set_random_policy();
            // Eigen::VectorXd pp = limbo::tools::random_vector(_policy.params().size()).array() * 2.0 * _boundary - _boundary;
            // Eigen::read_binary("policy_params_1.bin", pp);
            // _policy.set_params(pp);

            std::cout << "Executing random actions..." << std::endl;
            for (size_t i = 0; i < init; i++) {
                if (_random_policies) {
                    Eigen::VectorXd pp = limbo::tools::random_vector(_policy.params().size()).array() * 2.0 * _boundary - _boundary;
                    _policy.set_params(pp);
                }
                execute_and_record_data();
            }

            std::chrono::steady_clock::time_point time_start;
            std::cout << "Starting learning..." << std::endl;
            for (size_t i = 0; i < iterations; i++) {
                std::cout << std::endl
                          << "Learning iteration #" << (i + 1) << std::endl;

                time_start = std::chrono::steady_clock::now();
                learn_model();
                double learn_model_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - time_start).count();
                _ofs_model << learn_model_ms << std::endl;
                _model.save_data("gp_learn_" + std::to_string(i) + ".dat");

                _policy.normalize(_model);
                std::cout << "Learned model..." << std::endl;

                if (Params::verbose()) {
                    Eigen::VectorXd errors, sigmas;
                    std::tie(errors, sigmas) = get_accuracy();
                    std::cout << "Errors: " << errors.transpose() << std::endl;
                    std::cout << "Sigmas: " << sigmas.transpose() << std::endl;
                }

                time_start = std::chrono::steady_clock::now();
                optimize_policy(i + 1);
                double optimize_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - time_start).count();
                std::cout << "Optimized policy..." << std::endl;

                execute_and_record_data();
                std::cout << "Executed action..." << std::endl;
                std::cout << "Learning time: " << learn_model_ms << std::endl;
                std::cout << "Optimization time: " << optimize_ms << std::endl;
                _ofs_opt << optimize_ms << std::endl;
            }
            _ofs.close();
            _ofs_opt.close();
            _ofs_model.close();
            std::cout << "Experiment finished" << std::endl;
        }

    protected:
        Robot _robot;
        Policy _policy;
        Model _model;
        std::ofstream _ofs, _ofs_opt, _ofs_model;
        Eigen::VectorXd _params_starting;
        double _best;
        bool _random_policies;

        // state, action, prediction
        std::vector<std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd>> _observations;

        limbo::opt::eval_t _optimize_policy(const Eigen::VectorXd& params, bool eval_grad = false)
        {
            RewardFunction world;
            Policy policy;

            policy.set_params(params.array());
            policy.normalize(_model);

            // std::vector<double> R;
            // _robot.execute(policy, world, Params::medrops::rollout_steps(), R, false);
            //
            // double r = std::accumulate(R.begin(), R.end(), 0.0);
            double r = _robot.predict_policy(policy, _model, world, Params::medrops::rollout_steps());

            _opt_iters++;
            if (_max_reward < r) {
                // if (Params::verbose()) {
                //     std::vector<double> R;
                //     _robot.execute_dummy(policy, _model, world, Params::medrops::rollout_steps(), R, false);
                //     double simu_reward = std::accumulate(R.begin(), R.end(), 0.0);
                //     _robot.execute(policy, world, Params::medrops::rollout_steps(), R, false);
                //     double real_reward = std::accumulate(R.begin(), R.end(), 0.0);
                //
                //     _max_simu_reward = simu_reward;
                //     _max_real_reward = real_reward;
                // }
                _max_reward = r;
                _max_params = policy.params().array();
                // Eigen::write_binary("max_params.bin", policy.params());
            }

            // if (_opt_iters % 500 == 0)
            //     std::cout << _opt_iters << "(" << _max_reward << ") " << std::flush;

            // if (Params::verbose() && _opt_iters % 1000 == 0) {
            //     std::cout << _opt_iters << "(" << _max_reward << ", " << _max_simu_reward << ", " << _max_real_reward << ") " << std::flush;
            // }

            return limbo::opt::no_grad(r);
        }

        Eigen::VectorXd get_random_vector(size_t dim, Eigen::VectorXd bounds) const
        {
            Eigen::VectorXd rv = (limbo::tools::random_vector(dim).array() * 2 - 1);
            // rv(0) *= 3; rv(1) *= 5; rv(2) *= 6; rv(3) *= M_PI; rv(4) *= 10;
            return rv.cwiseProduct(bounds);
        }

        std::vector<Eigen::VectorXd> random_vectors(size_t dim, size_t q, Eigen::VectorXd bounds) const
        {
            std::vector<Eigen::VectorXd> result(q);
            for (size_t i = 0; i < q; i++) {
                result[i] = get_random_vector(dim, bounds);
            }
            return result;
        }

        std::tuple<Eigen::VectorXd, Eigen::VectorXd> get_accuracy(double perc = 0.75) const
        {
            // get data
            int sample_size = _observations.size();
            int training_size = perc * sample_size;
            int test_size = sample_size - training_size;
            std::vector<std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd>> observations = _observations;

            // shuffle data
            std::random_shuffle(observations.begin(), observations.end());

            std::vector<std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd>>::const_iterator first = observations.begin();
            std::vector<std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd>>::const_iterator last = observations.begin() + training_size;
            std::vector<std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd>> training_samples(first, last);

            first = observations.begin() + training_size;
            last = observations.end();
            std::vector<std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd>> test_samples(first, last);

            // create model
            Model model;
            model.learn(training_samples);

            // Get errors and sigmas
            Eigen::VectorXd errors(Params::model_pred_dim());
            Eigen::VectorXd sigmas(Params::model_pred_dim());

            for (size_t i = 0; i < test_samples.size(); i++) {
                Eigen::VectorXd st, act, pred;
                st = std::get<0>(test_samples[i]);
                act = std::get<1>(test_samples[i]);
                pred = std::get<2>(test_samples[i]);

                Eigen::VectorXd s(st.size() + act.size());
                s.head(st.size()) = st;
                s.tail(act.size()) = act;

                Eigen::VectorXd mu, sigma;
                std::tie(mu, sigma) = model.predictm(s);
                errors.array() += (mu - pred).norm();
                sigmas.array() += sigma.array();
            }

            errors.array() = errors.array() / double(test_size);
            sigmas.array() = sigmas.array() / double(test_size);

            return std::make_tuple(errors, sigmas);
        }

        // std::tuple<Eigen::VectorXd, Eigen::VectorXd> get_accuracy(int evaluations = 100000) const
        // {
        //     Eigen::VectorXd bounds(5);
        //     bounds << 3, 5, 6, M_PI, 10;
        //     std::vector<Eigen::VectorXd> rvs = random_vectors(5, evaluations, bounds);
        //
        //     Eigen::VectorXd errors(4);
        //     Eigen::VectorXd sigmas(4);
        //     for (size_t i = 0; i < rvs.size(); i++) {
        //         Eigen::VectorXd s;
        //         Eigen::VectorXd m;
        //         Eigen::VectorXd cm = comp_predict(rvs[i]);
        //         Eigen::VectorXd gp_query(6);
        //         gp_query.segment(0, 3) = rvs[i].segment(0, 3);
        //         gp_query(3) = std::cos(rvs[i](3));
        //         gp_query(4) = std::sin(rvs[i](3));
        //         gp_query(5) = rvs[i](4);
        //         std::tie(m, s) = _model.predictm(gp_query);
        //
        //         errors.array() += (m - cm).array().abs();
        //         sigmas.array() += s.array();
        //     }
        //
        //     return std::make_tuple(
        //         errors.array() / evaluations * 1.0,
        //         sigmas.array() / evaluations * 1.0);
        // }
        //
        // Eigen::VectorXd comp_predict(const Eigen::VectorXd& v) const
        // {
        //     double dt = 0.1, t = 0.0;
        //
        //     boost::numeric::odeint::runge_kutta4<std::vector<double>> ode_stepper;
        //     std::vector<double> state(4);
        //     state[0] = v(0);
        //     state[1] = v(1);
        //     state[2] = v(2);
        //     state[3] = v(3);
        //
        //     boost::numeric::odeint::integrate_const(ode_stepper,
        //         std::bind(&Medrops::dynamics, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, v(4)),
        //         state, t, dt, dt / 2.0);
        //
        //     Eigen::VectorXd new_state = Eigen::VectorXd::Map(state.data(), state.size());
        //     return (new_state.array() - v.segment(0, 4).array());
        // }
        //
        // void dynamics(const std::vector<double>& x, std::vector<double>& dx, double t, double u) const
        // {
        //     double l = 0.5, m = 0.5, M = 0.5, g = 9.82, b = 0.1;
        //
        //     dx[0] = x[1];
        //     dx[1] = (2 * m * l * std::pow(x[2], 2.0) * std::sin(x[3]) + 3 * m * g * std::sin(x[3]) * std::cos(x[3]) + 4 * u - 4 * b * x[1]) / (4 * (M + m) - 3 * m * std::pow(std::cos(x[3]), 2.0));
        //     dx[2] = (-3 * m * l * std::pow(x[2], 2.0) * std::sin(x[3]) * std::cos(x[3]) - 6 * (M + m) * g * std::sin(x[3]) - 6 * (u - b * x[1]) * std::cos(x[3])) / (4 * l * (m + M) - 3 * m * l * std::pow(std::cos(x[3]), 2.0));
        //     dx[3] = x[2];
        // }
    };
}

#endif
