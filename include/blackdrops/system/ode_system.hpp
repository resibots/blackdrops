#ifndef BLACKDROPS_SYSTEM_ODE_SYSTEM_HPP
#define BLACKDROPS_SYSTEM_ODE_SYSTEM_HPP

#include <boost/numeric/odeint.hpp>
#include <utils/utils.hpp>

namespace blackdrops {
    namespace system {
        template <typename Params>
        struct ODESystem {

            template <typename Policy, typename Reward>
            std::vector<std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd>> execute(const Policy& policy, const Reward& world, double T, std::vector<double>& R, bool display = true)
            {
                double dt = Params::blackdrops::dt();
                std::vector<std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd>> res;

                R = std::vector<double>();
                _last_states.clear();
                _last_commands.clear();

                Eigen::VectorXd init_diff = this->init_state();
                _last_states.push_back(init_diff);

                for (double t = 0.0; t <= T; t += dt) {
                    Eigen::VectorXd init = this->transform_state(init_diff);

                    Eigen::VectorXd u = policy.next(init);

                    std::vector<double> robot_state(init_diff.size(), 0.0);
                    Eigen::VectorXd::Map(robot_state.data(), robot_state.size()) = init_diff;

                    boost::numeric::odeint::integrate_const(boost::numeric::odeint::make_dense_output(1.0e-12, 1.0e-12, boost::numeric::odeint::runge_kutta_dopri5<std::vector<double>>()),
                        std::bind(&ODESystem::dynamics, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, u),
                        robot_state, t, t + dt, dt / 4.0);
                    Eigen::VectorXd final = Eigen::VectorXd::Map(robot_state.data(), robot_state.size());

                    if (display)
                        this->draw_single(final);

                    _last_states.push_back(final);
                    _last_commands.push_back(u);

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

            // get states from last execution
            std::vector<Eigen::VectorXd> get_last_states() const
            {
                return _last_states;
            }

            // get commands from last execution
            std::vector<Eigen::VectorXd> get_last_commands() const
            {
                return _last_commands;
            }

            virtual void draw_single(const Eigen::VectorXd& state) const {}

            virtual void dynamics(const std::vector<double>& x, std::vector<double>& dx, double t, const Eigen::VectorXd& u) const = 0;

        protected:
            std::vector<Eigen::VectorXd> _last_states, _last_commands;
        };
    }
}

#endif