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

#include <blackdrops/system/system.hpp>
#include <blackdrops/utils/utils.hpp>

namespace blackdrops {
    namespace system {
        template <typename Params, typename RolloutInfo>
        struct ODESystem : public System<Params, ODESystem<Params, RolloutInfo>, RolloutInfo> {

            template <typename Policy, typename Reward>
            std::vector<std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd>> execute(const Policy& policy, Reward& world, double T, std::vector<double>& R, bool display = true)
            {
                int H = std::ceil(T / Params::blackdrops::dt());
                std::vector<std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd>> res;

                R = std::vector<double>();
                if (display) {
                    this->_last_states.clear();
                    this->_last_commands.clear();
                }

                // Get the information of the rollout
                RolloutInfo rollout_info = this->get_rollout_info();

                Eigen::VectorXd init_true = rollout_info.init_state;
                Eigen::VectorXd init_diff = this->add_noise(init_true);
                if (display)
                    this->_last_states.push_back(init_diff);

                boost::numeric::odeint::runge_kutta_dopri5<std::vector<double>> _stepper;
                double t = 0.0;
                double dt = Params::blackdrops::dt();
                for (int i = 0; i < H; i++) {
                    Eigen::VectorXd init = this->transform_state(init_diff);

                    Eigen::VectorXd u = policy.next(this->policy_transform(init, &rollout_info));

                    std::vector<double> robot_state(init_true.size(), 0.0);
                    Eigen::VectorXd::Map(robot_state.data(), robot_state.size()) = init_true;

                    boost::numeric::odeint::integrate_const(boost::numeric::odeint::make_dense_output(1.0e-12, 1.0e-12, _stepper),
                        std::bind(&ODESystem::dynamics, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, u),
                        robot_state, t, t + dt, dt / 4.0);
                    t += dt;
                    Eigen::VectorXd final = Eigen::VectorXd::Map(robot_state.data(), robot_state.size());

                    if (display)
                        this->draw_single(final);

                    // add noise to our observation
                    Eigen::VectorXd obs = this->add_noise(final);

                    if (display) {
                        this->_last_states.push_back(obs);
                        this->_last_commands.push_back(u);
                    }

                    res.push_back(std::make_tuple(init, u, obs - init_diff));

                    // We want the actual reward of the system (i.e., with the noiseless states)
                    // this is not given to the algorithm
                    double r = world.observe(rollout_info, init_true, u, final, display);
                    R.push_back(r);

                    init_diff = obs;
                    init_true = final;
                    rollout_info.t += Params::blackdrops::dt();
                }

                if (!policy.random() && display) {
                    double rr = std::accumulate(R.begin(), R.end(), 0.0);
                    std::cout << "Reward: " << rr << std::endl;
                }

                return res;
            }

            virtual void draw_single(const Eigen::VectorXd& state) const {}

            virtual void dynamics(const std::vector<double>& x, std::vector<double>& dx, double t, const Eigen::VectorXd& u) const = 0;
        };
    } // namespace system
} // namespace blackdrops

#endif