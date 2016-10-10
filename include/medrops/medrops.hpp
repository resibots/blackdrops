#ifndef MEDROPS_MEDROPS_HPP
#define MEDROPS_MEDROPS_HPP

#include <limbo/opt/optimizer.hpp>
#include <fstream>

namespace medrops {

    template <typename Params, typename Model, typename Robot, typename Policy, typename PolicyOptimizer, typename RewardFunction>
    class Medrops {
    public:
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
            Eigen::VectorXd params_star = policy_optimizer(std::bind(&Medrops::_optimize_policy, this, std::placeholders::_1, std::placeholders::_2), _policy.params(), false);

            // std::cout << "BEST: " << limbo::opt::fun(_optimize_policy(params_star)) << std::endl;

            _policy.set_params(params_star);

            std::vector<double> R;
            RewardFunction world;
            _robot.execute_dummy(_policy, _model, world, Params::medrops::rollout_steps(), R);
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
                std::cout << "Learning iteration #" << (i + 1) << std::endl;
                learn_model();
                std::cout << "Learned model..." << std::endl;
                optimize_policy();
                std::cout << "Optimized policy..." << std::endl;
                execute_and_record_data();
                std::cout << "Executed action..." << std::endl;
            }
            _ofs.close();
        }

    protected:
        Robot _robot;
        Policy _policy;
        Model _model;
        std::ofstream _ofs;

        // state, action, prediction
        std::vector<std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd>> _observations;

        limbo::opt::eval_t _optimize_policy(const Eigen::VectorXd& params, bool eval_grad = false) const
        {
            RewardFunction world;
            Policy policy;
            policy.set_params(params);

            double r = _robot.predict_policy(policy, _model, world, Params::medrops::rollout_steps());

            return limbo::opt::no_grad(r);
        }
    };
}

#endif
