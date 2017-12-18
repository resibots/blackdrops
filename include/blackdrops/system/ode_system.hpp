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
//| This software is the implementation of the Black-DROPS algorithm, which is
//| a model-based policy search algorithm with the following main properties:
//|   - uses Gaussian processes (GPs) to model the dynamics of the robot/system
//|   - takes into account the uncertainty of the dynamical model when
//|                                                      searching for a policy
//|   - is data-efficient or sample-efficient; i.e., it requires very small
//|     interaction time with the system to find a working policy (e.g.,
//|     around 16-20 seconds to learn a policy for the cart-pole swing up task)
//|   - when several cores are available, it can be faster than analytical
//|                                                    approaches (e.g., PILCO)
//|   - it imposes no constraints on the type of the reward function (it can
//|                                                  also be learned from data)
//|   - it imposes no constraints on the type of the policy representation
//|     (any parameterized policy can be used --- e.g., dynamic movement
//|                                              primitives or neural networks)
//|
//| Main repository: http://github.com/resibots/blackdrops
//| Preprint: https://arxiv.org/abs/1703.07261
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
                int H = std::ceil(Params::blackdrops::T() / Params::blackdrops::dt());
                std::vector<std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd>> res;

                R = std::vector<double>();
                _last_states.clear();
                _last_commands.clear();

                Eigen::VectorXd init_diff = this->add_noise(this->init_state());
                _last_states.push_back(init_diff);

                double t = 0.0;
                double dt = Params::blackdrops::dt();
                for (int i = 0; i < H; i++) {
                    Eigen::VectorXd init = this->transform_state(init_diff);

                    Eigen::VectorXd u = policy.next(init);

                    std::vector<double> robot_state(init_diff.size(), 0.0);
                    Eigen::VectorXd::Map(robot_state.data(), robot_state.size()) = init_diff;

                    boost::numeric::odeint::integrate_const(boost::numeric::odeint::make_dense_output(1.0e-12, 1.0e-12, boost::numeric::odeint::runge_kutta_dopri5<std::vector<double>>()),
                        std::bind(&ODESystem::dynamics, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, u),
                        robot_state, t, t + dt, dt / 4.0);
                    t += dt;
                    Eigen::VectorXd final = Eigen::VectorXd::Map(robot_state.data(), robot_state.size());

                    if (display)
                        this->draw_single(final);

                    // add noise to our observation
                    Eigen::VectorXd obs = this->add_noise(final);

                    _last_states.push_back(obs);
                    _last_commands.push_back(u);

                    res.push_back(std::make_tuple(init, u, obs - init_diff));
                    double r = world(init, u, obs);
                    R.push_back(r);

                    init_diff = obs;
                }

                if (!policy.random() && display) {
                    double rr = std::accumulate(R.begin(), R.end(), 0.0);
                    std::cout << "Reward: " << rr << std::endl;
                }

                return res;
            }

            template <typename Policy, typename Model, typename Reward>
            void execute_dummy(const Policy& policy, const Model& model, const Reward& world, double T, std::vector<double>& R, bool display = true)
            {
                std::vector<Eigen::VectorXd> states, commands;

                int H = std::ceil(Params::blackdrops::T() / Params::blackdrops::dt());
                R = std::vector<double>();
                // init state
                Eigen::VectorXd init_diff = this->init_state();

                Eigen::VectorXd init = this->transform_state(init_diff);

                states.push_back(init_diff);

                for (int i = 0; i < H; i++) {
                    Eigen::VectorXd query_vec(Params::blackdrops::model_input_dim() + Params::blackdrops::action_dim());

                    Eigen::VectorXd u = policy.next(init);
                    query_vec.head(Params::blackdrops::model_input_dim()) = init;
                    query_vec.tail(Params::blackdrops::action_dim()) = u;

                    commands.push_back(u);

                    Eigen::VectorXd mu;
                    Eigen::VectorXd sigma;
                    std::tie(mu, sigma) = model.predictm(query_vec);

                    Eigen::VectorXd final = init_diff + mu;

                    states.push_back(final);

                    double r = world(init_diff, mu, final);
                    R.push_back(r);

                    init_diff = final;
                    init = this->transform_state(init_diff);
                }

                _last_dummy_states = states;
                _last_dummy_commands = commands;
            }

            template <typename Policy, typename Model, typename Reward>
            double predict_policy(const Policy& policy, const Model& model, const Reward& world, double T) const
            {
                int H = std::ceil(Params::blackdrops::T() / Params::blackdrops::dt());
                double reward = 0.0;
                // init state
                Eigen::VectorXd init_diff = this->init_state();

                Eigen::VectorXd init = this->transform_state(init_diff);

                for (int i = 0; i < H; i++) {
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

            // add noise to the observed state if desired
            // by default, no noise is added
            virtual Eigen::VectorXd add_noise(const Eigen::VectorXd& original_state) const
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

            // get states from last dummy execution
            std::vector<Eigen::VectorXd> get_last_dummy_states() const
            {
                return _last_dummy_states;
            }

            // get commands from lastd ummy execution
            std::vector<Eigen::VectorXd> get_last_dummy_commands() const
            {
                return _last_dummy_commands;
            }

            virtual void draw_single(const Eigen::VectorXd& state) const {}

            virtual void dynamics(const std::vector<double>& x, std::vector<double>& dx, double t, const Eigen::VectorXd& u) const = 0;

        protected:
            std::vector<Eigen::VectorXd> _last_states, _last_commands;
            std::vector<Eigen::VectorXd> _last_dummy_states, _last_dummy_commands;
        };
    } // namespace system
} // namespace blackdrops

#endif