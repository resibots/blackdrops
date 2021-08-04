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
#ifndef BLACKDROPS_SYSTEM_DART_SYSTEM_HPP
#define BLACKDROPS_SYSTEM_DART_SYSTEM_HPP

#include <functional>

#include <robot_dart/control/robot_control.hpp>
#include <robot_dart/robot.hpp>
#include <robot_dart/robot_dart_simu.hpp>
#include <robot_dart/utils.hpp>

#include <dart/dynamics/BodyNode.hpp>

#ifdef GRAPHIC
#include <robot_dart/gui/magnum/graphics.hpp>
#endif

#include <blackdrops/system/system.hpp>
#include <blackdrops/utils/utils.hpp>

namespace blackdrops {
    namespace system {
        template <typename Params, typename PolicyController, typename RolloutInfo>
        struct DARTSystem : public System<Params, DARTSystem<Params, PolicyController, RolloutInfo>, RolloutInfo> {
            using robot_simu_t = robot_dart::RobotDARTSimu;

            template <typename Policy, typename Reward>
            std::vector<std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd>> execute(const Policy& policy, Reward& world, double T, std::vector<double>& R, bool display = true)
            {
                // Make sure that the simulation step is smaller than the sampling/control rate
                assert(Params::dart_system::sim_step() < Params::blackdrops::dt());

                std::vector<std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd>> res;

                R = std::vector<double>();

                robot_simu_t simu;
#ifdef GRAPHIC
                simu.set_graphics(std::make_shared<robot_dart::gui::magnum::Graphics>());
                simu.graphics()->set_enable(display);
#endif
                // simulation step different from sampling rate -- we need a stable simulation
                simu.set_timestep(Params::dart_system::sim_step());

                // Get the information of the rollout
                RolloutInfo rollout_info = this->get_rollout_info();

                // setup robot
                std::shared_ptr<robot_dart::Robot> simulated_robot = this->get_robot();
                simulated_robot->set_actuator_types(Params::dart_policy_control::joint_type());

                // setup the controller
                auto controller = std::make_shared<PolicyController>(policy.params());
                controller->set_transform_state(std::bind(&DARTSystem::transform_state, this, std::placeholders::_1));
                controller->set_noise_function(std::bind(&DARTSystem::add_noise, this, std::placeholders::_1));
                controller->set_update_function(std::bind([&](double t) { rollout_info.t = t; }, std::placeholders::_1));
                controller->set_policy_function(std::bind(&DARTSystem::policy_transform, this, std::placeholders::_1, &rollout_info));

                // add the controller to the robot
                simulated_robot->add_controller(controller);
                // add the robot to the simulation
                simu.add_robot(simulated_robot);

                // Add extra to simu object
                this->add_extra_to_simu(simu, rollout_info);

                // Get initial state from info and add noise
                Eigen::VectorXd init_diff = rollout_info.init_state;
                this->set_robot_state(simulated_robot, init_diff);
                init_diff = this->add_noise(init_diff);

                simu.run(T + Params::dart_system::sim_step());

                std::vector<Eigen::VectorXd> states = controller->get_states();
                std::vector<Eigen::VectorXd> noiseless_states = controller->get_noiseless_states();
                if (display)
                    this->_last_states = states;
                std::vector<Eigen::VectorXd> commands = controller->get_commands();
                if (display)
                    this->_last_commands = commands;

                for (size_t j = 0; j < states.size() - 1; j++) {
                    Eigen::VectorXd init = states[j];

                    Eigen::VectorXd init_full = this->transform_state(init);

                    Eigen::VectorXd u = commands[j];
                    Eigen::VectorXd final = states[j + 1];

                    // We want the actual reward of the system (i.e., with the noiseless states)
                    // this is not given to the algorithm
                    double r = world.observe(rollout_info, noiseless_states[j], u, noiseless_states[j + 1], display);
                    R.push_back(r);
                    res.push_back(std::make_tuple(init_full, u, final - init));
                }

                if (!policy.random() && display) {
                    double rr = std::accumulate(R.begin(), R.end(), 0.0);
                    std::cout << "Reward: " << rr << std::endl;
                }

                return res;
            }

            // override this to add extra stuff to the robot_dart simulator
            virtual void add_extra_to_simu(robot_simu_t& simu, const RolloutInfo& rollout_info) const {}

            // you should override this, to define how your simulated robot_dart::Robot will be constructed
            virtual std::shared_ptr<robot_dart::Robot> get_robot() const = 0;

            // override this if you want to set in a specific way the initial state of your robot
            virtual void set_robot_state(const std::shared_ptr<robot_dart::Robot>& robot, const Eigen::VectorXd& state) const {}
        };

        template <typename Params, typename Policy>
        class BaseDARTPolicyControl : public robot_dart::control::RobotControl {
        public:
            using robot_t = std::shared_ptr<robot_dart::Robot>;

            BaseDARTPolicyControl() {}
            BaseDARTPolicyControl(const Eigen::VectorXd& ctrl, bool full_control = false)
                : robot_dart::control::RobotControl(ctrl, full_control)
            {
                // set some default functions in case the user does not define them
                set_transform_state(std::bind(&BaseDARTPolicyControl::transform_state, this, std::placeholders::_1));
                set_noise_function(std::bind(&BaseDARTPolicyControl::transform_state, this, std::placeholders::_1));
                set_policy_function(std::bind(&BaseDARTPolicyControl::transform_state, this, std::placeholders::_1));
                set_update_function(std::bind(&BaseDARTPolicyControl::dummy, this, std::placeholders::_1));
            }

            void configure() override
            {
                _prev_time = 0.0;
                _t = 0.0;
                _first = true;

                _policy.set_params(_ctrl);

                _states.clear();
                _noiseless_states.clear();
                _coms.clear();

                if (Params::blackdrops::action_dim() == _control_dof)
                    _active = true;
            }

            Eigen::VectorXd calculate(double t) override
            {
                _t = t;
                _update_func(t);

                double dt = Params::blackdrops::dt();

                if (_first || (_t - _prev_time - dt) > -Params::dart_system::sim_step() / 2.0) {
                    Eigen::VectorXd q = this->get_state(_robot.lock());
                    _noiseless_states.push_back(q);
                    q = _add_noise(q);
                    Eigen::VectorXd commands = _policy.next(_policy_state(_tranform_state(q)));
                    _states.push_back(q);
                    _coms.push_back(commands);

                    ROBOT_DART_ASSERT(_control_dof == commands.size(), "BaseDARTPolicyControl: Policy output size is not the same as the control DOFs of the robot", Eigen::VectorXd::Zero(_control_dof));
                    _prev_commands = commands;
                    _prev_time = _t;
                    _first = false;
                }

                return _prev_commands;
            }

            std::vector<Eigen::VectorXd> get_states() const
            {
                return _states;
            }

            std::vector<Eigen::VectorXd> get_noiseless_states() const
            {
                return _noiseless_states;
            }

            std::vector<Eigen::VectorXd> get_commands() const
            {
                return _coms;
            }

            void set_transform_state(std::function<Eigen::VectorXd(const Eigen::VectorXd&)> func)
            {
                _tranform_state = func;
            }

            void set_noise_function(std::function<Eigen::VectorXd(const Eigen::VectorXd&)> func)
            {
                _add_noise = func;
            }

            void set_policy_function(std::function<Eigen::VectorXd(const Eigen::VectorXd&)> func)
            {
                _policy_state = func;
            }

            void set_update_function(std::function<void(double)> func)
            {
                _update_func = func;
            }

            virtual Eigen::VectorXd get_state(const robot_t& robot) const = 0;

        protected:
            double _prev_time;
            double _t;
            bool _first;
            Eigen::VectorXd _prev_commands;
            Policy _policy;
            std::vector<Eigen::VectorXd> _coms;
            std::vector<Eigen::VectorXd> _states, _noiseless_states;
            std::function<Eigen::VectorXd(const Eigen::VectorXd&)> _tranform_state;
            std::function<Eigen::VectorXd(const Eigen::VectorXd&)> _add_noise;
            std::function<Eigen::VectorXd(const Eigen::VectorXd&)> _policy_state;
            std::function<void(double)> _update_func;

            Eigen::VectorXd transform_state(const Eigen::VectorXd& original_state) const
            {
                return original_state;
            }

            void dummy(double) const {}
        };
    } // namespace system
} // namespace blackdrops

#endif