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
//| This software is a computer library whose purpose is to optimize continuous,
//| black-box functions. It mainly implements Gaussian processes and Bayesian
//| optimization.
//| Main repository: http://github.com/resibots/blackdrops
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
#ifndef BLACKDROPS_BLACKDROPS_HPP
#define BLACKDROPS_BLACKDROPS_HPP

#include <Eigen/binary_matrix.hpp>
#include <chrono>
#include <fstream>
#include <limbo/opt/optimizer.hpp>
#include <limits>

namespace blackdrops {

    template <typename Params, typename Model, typename Robot, typename Policy, typename PolicyOptimizer, typename RewardFunction>
    class BlackDROPS {
    public:
        BlackDROPS() : _best(-std::numeric_limits<double>::max()) {}
        ~BlackDROPS() {}

        void execute_and_record_data()
        {
            std::vector<double> R;
            RewardFunction world;
            // Execute best policy so far on robot
            auto obs_new = _robot.execute(_policy, world, Params::blackdrops::T(), R);

            // Check if it is better than the previous best
            double r_new = std::accumulate(R.begin(), R.end(), 0.0);
            if (r_new > _best) {
                _best = r_new;
                _params_starting = _policy.params();
            }

            // Append recorded data
            _observations.insert(_observations.end(), obs_new.begin(), obs_new.end());

            // statistics for immediate rewards
            for (auto r : R)
                _ofs_real << r << " ";
            _ofs_real << std::endl;

            // statistics for cumulative reward
            double r = std::accumulate(R.begin(), R.end(), 0.0);
            _ofs_results << r << std::endl;

            // statistics for trajectories
            std::vector<Eigen::VectorXd> states = _robot.get_last_states();
            std::vector<Eigen::VectorXd> commands = _robot.get_last_commands();

            for (size_t i = 0; i < R.size(); i++) {
                Eigen::VectorXd state = states[i];
                Eigen::VectorXd command = commands[i];

                for (int j = 0; j < state.size(); j++)
                    _ofs_traj_real << state(j) << " ";
                for (int j = 0; j < command.size(); j++)
                    _ofs_traj_real << command(j) << " ";
                _ofs_traj_real << std::endl;
            }
            Eigen::VectorXd state = states.back();
            for (int j = 0; j < state.size(); j++)
                _ofs_traj_real << state(j) << " ";
            for (int j = 0; j < commands.back().size(); j++)
                _ofs_traj_real << "0.0 ";
            _ofs_traj_real << std::endl;
        }

        void learn_model()
        {
            _model.learn(_observations);
        }

        void optimize_policy(size_t i)
        {
            PolicyOptimizer policy_optimizer;

            Eigen::VectorXd params_star;
            Eigen::VectorXd params_starting = _policy.params();
            if (_random_policies)
                params_starting = _params_starting;
            Eigen::write_binary("policy_params_starting_" + std::to_string(i) + ".bin", params_starting);

            _opt_iters = 0;
            _max_reward = -std::numeric_limits<double>::max();
            if (_boundary == 0) {
                std::cout << "Optimizing policy... " << std::flush;
                params_star = policy_optimizer(
                    std::bind(&BlackDROPS::_optimize_policy, this, std::placeholders::_1, std::placeholders::_2),
                    params_starting,
                    false);
            }
            else {
                std::cout << "Optimizing policy bounded to [-" << _boundary << ", " << _boundary << "]... " << std::flush;
                params_star = policy_optimizer(
                    std::bind(&BlackDROPS::_optimize_policy, this, std::placeholders::_1, std::placeholders::_2),
                    params_starting,
                    true);
            }
            if (Params::blackdrops::verbose())
                std::cout << _opt_iters << "(" << _max_reward << ")" << std::endl;
            else
                std::cout << std::endl;
            std::cout << "Optimization iterations: " << _opt_iters << std::endl;

            // Since we are optimizing a noisy function, it is not good to keep the best ever found
            // if (Params::opt_cmaes::elitism() == 0)
            //     params_star = _max_params;

            _policy.set_params(params_star);

            Eigen::write_binary("policy_params_" + std::to_string(i) + ".bin", _policy.params());

            std::vector<double> R;
            RewardFunction world;
            _robot.execute_dummy(_policy, _model, world, Params::blackdrops::T(), R);
            std::cout << "Dummy reward: " << std::accumulate(R.begin(), R.end(), 0.0) << std::endl;

            _ofs_traj_dummy.open("traj_dummy_" + std::to_string(i) + ".dat");
            // statistics for trajectories
            std::vector<Eigen::VectorXd> states = _robot.get_last_dummy_states();
            std::vector<Eigen::VectorXd> commands = _robot.get_last_dummy_commands();

            for (size_t i = 0; i < R.size(); i++) {
                Eigen::VectorXd state = states[i];
                Eigen::VectorXd command = commands[i];

                for (int j = 0; j < state.size(); j++)
                    _ofs_traj_dummy << state(j) << " ";
                for (int j = 0; j < command.size(); j++)
                    _ofs_traj_dummy << command(j) << " ";
                _ofs_traj_dummy << std::endl;
            }
            Eigen::VectorXd state = states.back();
            for (int j = 0; j < state.size(); j++)
                _ofs_traj_dummy << state(j) << " ";
            for (int j = 0; j < commands.back().size(); j++)
                _ofs_traj_dummy << "0.0 ";
            _ofs_traj_dummy << std::endl;
            _ofs_traj_dummy.close();

            for (auto r : R)
                _ofs_esti << r << " ";
            _ofs_esti << std::endl;
        }

        void learn(size_t init, size_t iterations, bool random_policies = false, const std::string& policy_file = "")
        {
            _boundary = Params::blackdrops::boundary();
            _random_policies = random_policies;
            // TO-DO: add prefix
            _ofs_results.open("results.dat");
            _ofs_real.open("real.dat");
            _ofs_esti.open("estimates.dat");
            _ofs_opt.open("times.dat");
            _ofs_model.open("times_model.dat");
            _policy.set_random_policy();
            _best = -std::numeric_limits<double>::max();

#ifdef MEAN
            _random_policies = true;
            if (policy_file == "") {
                Eigen::VectorXd pp = limbo::tools::random_vector(_policy.params().size()).array() * 2.0 * _boundary - _boundary;
                _policy.set_params(pp);
                _params_starting = pp;
                optimize_policy(0);
                _params_starting = _policy.params();
                optimize_policy(0);
            }
            else {
                Eigen::VectorXd params;
                Eigen::read_binary(policy_file, params);
                _policy.set_params(params);
            }
            _ofs_traj_real.open("traj_real_0.dat");
            execute_and_record_data();
            _ofs_traj_real.close();
#else

            std::cout << "Executing random actions..." << std::endl;
            for (size_t i = 0; i < init; i++) {
                _ofs_traj_real.open("traj_real_" + std::to_string(i) + ".dat");
                if (_random_policies) {
                    Eigen::VectorXd pp = limbo::tools::random_vector(_policy.params().size()).array() * 2.0 * _boundary - _boundary;
                    _policy.set_params(pp);
                    Eigen::write_binary("random_policy_params_" + std::to_string(i) + ".bin", pp);
                }
                execute_and_record_data();
                _ofs_traj_real.close();
            }
#endif

            std::chrono::steady_clock::time_point time_start;
            std::cout << "Starting learning..." << std::endl;
            for (size_t i = 0; i < iterations; i++) {
                _ofs_traj_real.open("traj_real_" + std::to_string(i + init) + ".dat");
                std::cout << std::endl
                          << "Learning iteration #" << (i + 1) << std::endl;

                time_start = std::chrono::steady_clock::now();
                learn_model();
                double learn_model_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - time_start).count();
                _ofs_model << (learn_model_ms * 1e-3) << std::endl;
                _model.save_data("gp_learn_" + std::to_string(i) + ".dat");

                std::cout << "Learned model..." << std::endl;
                std::cout << "Learning time: " << learn_model_ms * 1e-3 << "s" << std::endl;

                if (Params::blackdrops::verbose()) {
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
                std::cout << "Executed policy..." << std::endl;
                std::cout << "Optimization time: " << optimize_ms * 1e-3 << "s" << std::endl;
                _ofs_opt << (optimize_ms * 1e-3) << std::endl;
                _ofs_traj_real.close();
            }
            _ofs_real.close();
            _ofs_esti.close();
            _ofs_opt.close();
            _ofs_model.close();
            _ofs_results.close();
            std::cout << "Experiment finished" << std::endl;
        }

    protected:
        Robot _robot;
        Policy _policy;
        Model _model;
        std::ofstream _ofs_real, _ofs_esti, _ofs_traj_real, _ofs_traj_dummy, _ofs_results, _ofs_opt, _ofs_model;
        Eigen::VectorXd _params_starting;
        double _best;
        bool _random_policies;
        int _opt_iters;
        double _max_reward;
        Eigen::VectorXd _max_params;
        double _boundary;
        std::mutex _iter_mutex;

        // state, action, prediction
        std::vector<std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd>> _observations;

        limbo::opt::eval_t _optimize_policy(const Eigen::VectorXd& params, bool eval_grad = false)
        {
            RewardFunction world;
            Policy policy;

            policy.set_params(params.array());

            // std::vector<double> R;
            // _robot.execute(policy, world, Params::blackdrops::T(), R, false);

            // double r = std::accumulate(R.begin(), R.end(), 0.0);
            double r = _robot.predict_policy(policy, _model, world, Params::blackdrops::T());

            _iter_mutex.lock();
            _opt_iters++;
            _iter_mutex.unlock();
            if (_max_reward < r) {
                _max_reward = r;
                _max_params = policy.params().array();
                if (Params::blackdrops::verbose())
                    std::cout << "(" << _opt_iters << ", " << _max_reward << "), " << std::flush;
            }

            return limbo::opt::no_grad(r);
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
            Eigen::VectorXd errors = Eigen::VectorXd::Zero(Params::blackdrops::model_pred_dim());
            Eigen::VectorXd sigmas = Eigen::VectorXd::Zero(Params::blackdrops::model_pred_dim());

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
                errors.array() += (mu - pred).array().square().sqrt();
                sigmas.array() += sigma.array();
            }

            errors.array() = errors.array() / double(test_size);
            sigmas.array() = sigmas.array() / double(test_size);

            return std::make_tuple(errors, sigmas);
        }
    };
}

#endif
