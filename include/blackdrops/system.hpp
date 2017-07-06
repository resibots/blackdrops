#ifndef BLACKDROPS_SYSTEM_HPP
#define BLACKDROPS_SYSTEM_HPP

#include <utils/utils.hpp>

namespace blackdrops {
    template <typename Params>
    struct System {

        template <typename Policy, typename Reward>
        std::vector<std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd>> execute(const Policy& policy, const Reward& world, double T, std::vector<double>& R, bool display = true) const
        {
            double dt = Params::blackdrops::dt();
            std::vector<std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd>> res;

            R = std::vector<double>();

            Eigen::VectorXd init_diff = this->init_state();

            for (double t = 0.0; t <= T; t += dt) {
                Eigen::VectorXd init = this->transform_state(init_diff);

                Eigen::VectorXd u = policy.next(init);

                Eigen::VectorXd final = this->execute_single(init_diff, u, t, display);

                res.push_back(std::make_tuple(init, u, final - init_diff));
                double r = world(init, u, final);
                R.push_back(r);

                init_diff = final;
            }

            if (!policy.random() && display) {
                double rr = std::accumulate(R.begin(), R.end(), 0.0);
                std::cout << "Reward: " << rr << std::endl;
            }

            return res;
        }

        template <typename Policy, typename Model, typename Reward>
        void execute_dummy(const Policy& policy, const Model& model, const Reward& world, double T, std::vector<double>& R, bool display = true) const
        {
            double dt = Params::blackdrops::dt();
            R = std::vector<double>();
            // init state
            Eigen::VectorXd init_diff = this->init_state();

            Eigen::VectorXd init = this->transform_state(init_diff);

            for (double t = 0.0; t <= T; t += dt) {
                Eigen::VectorXd query_vec(Params::blackdrops::model_input_dim() + Params::blackdrops::action_dim());
                Eigen::VectorXd u = policy.next(init);
                query_vec.head(Params::blackdrops::model_input_dim()) = init;
                query_vec.tail(Params::blackdrops::action_dim()) = u;

                Eigen::VectorXd mu;
                Eigen::VectorXd sigma;
                std::tie(mu, sigma) = model.predictm(query_vec);

                Eigen::VectorXd final = init_diff + mu;

                double r = world(init_diff, mu, final);
                R.push_back(r);

                init_diff = final;
                init = this->transform_state(init_diff);
            }
        }

        template <typename Policy, typename Model, typename Reward>
        double predict_policy(const Policy& policy, const Model& model, const Reward& world, double T) const
        {
            double dt = Params::blackdrops::dt();
            double reward = 0.0;
            // init state
            Eigen::VectorXd init_diff = this->init_state();

            Eigen::VectorXd init = this->transform_state(init_diff);

            for (double t = 0.0; t <= T; t += dt) {
                Eigen::VectorXd query_vec(Params::blackdrops::model_input_dim() + Params::blackdrops::action_dim());
                Eigen::VectorXd u = policy.next(init);
                query_vec.head(Params::blackdrops::model_input_dim()) = init;
                query_vec.tail(Params::blackdrops::action_dim()) = u;

                Eigen::VectorXd mu;
                Eigen::VectorXd sigma;
                std::tie(mu, sigma) = model.predictm(query_vec);

                if (Params::opt_cmaes::handle_uncertainty()) {
                    sigma = sigma.array().sqrt();
                    for (int i = 0; i < mu.size(); i++) {
                        double s = gaussian_rand(mu(i), sigma(i));
                        mu(i) = std::max(mu(i) - sigma(i),
                            std::min(s, mu(i) + sigma(i)));
                    }
                }

                Eigen::VectorXd final = init_diff + mu;

                reward += world(init_diff, u, final);
                init_diff = final;
                init = this->transform_state(init_diff);
            }

            return reward;
        }

        // transform the state input to the GPs and policy if needed
        // by default, no transformation is applied
        virtual Eigen::VectorXd transform_state(const Eigen::VectorXd& original_state) const
        {
            return original_state;
        }

        // return the initial state of the system
        // by default, the zero state is returned
        virtual Eigen::VectorXd init_state() const
        {
            return Eigen::VectorXd::Zero(Params::blackdrops::model_pred_dim());
        }

        virtual Eigen::VectorXd execute_single(const Eigen::VectorXd& state, const Eigen::VectorXd& u, double t, bool display = true) const = 0;
    };
}

#endif