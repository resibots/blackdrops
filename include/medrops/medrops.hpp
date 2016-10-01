#ifndef MEDROPS_MEDROPS_HPP
#define MEDROPS_MEDROPS_HPP

#include <limbo/opt/optimizer.hpp>

namespace medrops {

    template <typename Params, typename Model, typename Robot, typename Policy, typename PolicyOptimizer, typename RewardFunction>
    class Medrops {
    public:
        Medrops() {}
        ~Medrops() {}

        void execute_and_record_data()
        {
            // Execute best policy so far on robot
            auto obs_new = _robot.execute(_policy, Params::medrops::rollout_steps());
            // Append recorded data
            _observations.insert(_observations.end(), obs_new.begin(), obs_new.end());
        }

        void learn_model()
        {
            _model.learn(_observations);
        }

        void optimize_policy()
        {
            PolicyOptimizer policy_optimizer;

            // For now optimize policy without gradients
            Eigen::VectorXd params_star = policy_optimizer(std::bind(&Medrops::_optimize_policy, *this, std::placeholders::_1, std::placeholders::_2), _policy.params(), false);

            _policy.set_params(params_star);

            _robot.execute_dummy(_policy, _model, Params::medrops::rollout_steps());
        }

        void learn(size_t init, size_t iterations)
        {
            _policy.set_random_policy();

            std::cout << "Executing random actions..." << std::endl;
            for (size_t i = 0; i < init; i++) {
                execute_and_record_data();
            }

            std::cout << "Starting learning..." << std::endl;
            for (size_t i = 0; i < iterations; i++) {
                learn_model();
                std::cout << "Learned model..." << std::endl;
                optimize_policy();
                std::cout << "Optimized policy..." << std::endl;
                execute_and_record_data();
            }
        }

    protected:
        Robot _robot;
        Policy _policy;
        Model _model;

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
