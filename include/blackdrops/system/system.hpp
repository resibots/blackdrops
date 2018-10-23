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
#ifndef BLACKDROPS_SYSTEM_SYSTEM_HPP
#define BLACKDROPS_SYSTEM_SYSTEM_HPP

#include <blackdrops/utils/utils.hpp>

namespace blackdrops {
    namespace system {
        template <typename Params, typename MySystem, typename RolloutInfo>
        struct System {
            template <typename Policy, typename Reward>
            std::vector<std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd>> execute(const Policy& policy, Reward& world, double T, std::vector<double>& R, bool display = true)
            {
                return static_cast<MySystem*>(this)->execute(policy, world, T, R, display);
            }

            template <typename Policy, typename Model, typename Reward>
            void execute_dummy(const Policy& policy, const Model& model, const Reward& world, double T, std::vector<double>& R, bool display = true)
            {
                std::vector<Eigen::VectorXd> states, commands;

                int H = std::ceil(T / Params::blackdrops::dt());
                R = std::vector<double>();

                // Get the information of the rollout
                RolloutInfo rollout_info = get_rollout_info();

                // Get initial state from info
                Eigen::VectorXd init_diff = rollout_info.init_state;

                Eigen::VectorXd init = this->transform_state(init_diff);

                states.push_back(init_diff);

                for (int i = 0; i < H; i++) {
                    Eigen::VectorXd query_vec(Params::blackdrops::model_input_dim() + Params::blackdrops::action_dim());

                    Eigen::VectorXd u = policy.next(this->policy_transform(init, &rollout_info));
                    query_vec.head(Params::blackdrops::model_input_dim()) = init;
                    query_vec.tail(Params::blackdrops::action_dim()) = u;

                    commands.push_back(u);

                    Eigen::VectorXd mu;
                    Eigen::VectorXd sigma;
                    std::tie(mu, sigma) = model.predict(query_vec, false);

                    Eigen::VectorXd final = init_diff + mu;

                    states.push_back(final);

                    double r = world.query(rollout_info, init_diff, mu, final);
                    R.push_back(r);

                    init_diff = final;
                    init = this->transform_state(init_diff);
                    rollout_info.t += Params::blackdrops::dt();
                }

                _last_dummy_states = states;
                _last_dummy_commands = commands;
            }

            template <typename Policy, typename Model, typename Reward>
            double predict_policy(const Policy& policy, const Model& model, const Reward& world, double T) const
            {
                // Get the information of the rollout
                RolloutInfo rollout_info = get_rollout_info();

                std::vector<double> R;
                std::tie(std::ignore, std::ignore, R) = predict_policy(rollout_info.init_state, rollout_info, policy, model, world, T, Params::blackdrops::stochastic());

                return std::accumulate(R.begin(), R.end(), 0.0);
            }

            template <typename Policy, typename Model, typename Reward>
            std::tuple<std::vector<Eigen::VectorXd>, std::vector<Eigen::VectorXd>, std::vector<double>> predict_policy(const Eigen::VectorXd& init_state, RolloutInfo& rollout_info, const Policy& policy, const Model& model, const Reward& world, double T, bool with_variance = false) const
            {
                int H = std::ceil(T / Params::blackdrops::dt());
                std::vector<double> R;
                std::vector<Eigen::VectorXd> states, actions;

                // Set initial state
                Eigen::VectorXd init_diff = init_state;
                // Log initial state
                states.push_back(init_diff);

                Eigen::VectorXd init = this->transform_state(init_diff);

                for (int i = 0; i < H; i++) {
                    Eigen::VectorXd query_vec(Params::blackdrops::model_input_dim() + Params::blackdrops::action_dim());
                    Eigen::VectorXd u = policy.next(this->policy_transform(init, &rollout_info));
                    query_vec.head(Params::blackdrops::model_input_dim()) = init;
                    query_vec.tail(Params::blackdrops::action_dim()) = u;

                    Eigen::VectorXd mu;
                    Eigen::VectorXd sigma;
                    std::tie(mu, sigma) = model.predict(query_vec, with_variance);

                    if (with_variance) {
                        sigma = sigma.array().sqrt();
                        for (int i = 0; i < mu.size(); i++) {
                            double s = utils::gaussian_rand(mu(i), sigma(i));
                            mu(i) = std::max(mu(i) - sigma(i),
                                std::min(s, mu(i) + sigma(i)));
                        }
                    }

                    Eigen::VectorXd final = init_diff + mu;
                    states.push_back(final);
                    actions.push_back(u);

                    R.push_back(world.query(rollout_info, init_diff, u, final));
                    init_diff = final;
                    init = this->transform_state(init_diff);
                    rollout_info.t += Params::blackdrops::dt();
                }

                return std::make_tuple(states, actions, R);
            }

            // get information for rollout (i.e., initial state, target, etc.)
            // this is useful if you wish to generate some different conditions
            // that are constant throughout the same rollout, but different in different rollouts
            // by default, we only get the initial state
            virtual RolloutInfo get_rollout_info() const
            {
                RolloutInfo info;
                info.init_state = this->init_state();
                info.t = 0.;

                return info;
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

            // transform the state variables that go to the policy if needed
            // by default, no transformation is applied
            virtual Eigen::VectorXd policy_transform(const Eigen::VectorXd& original_state, RolloutInfo* info) const
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

        protected:
            std::vector<Eigen::VectorXd> _last_states, _last_commands;
            std::vector<Eigen::VectorXd> _last_dummy_states, _last_dummy_commands;
        };
    } // namespace system
} // namespace blackdrops

#endif