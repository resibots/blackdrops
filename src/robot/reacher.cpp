#include <future>
#include <limbo/limbo.hpp>

#include <trac_ik/trac_ik.hpp>

#include <boost/program_options.hpp>

#include <robot_dart/robot_dart_simu.hpp>

#ifdef GRAPHIC
#include <robot_dart/graphics.hpp>
#endif

#ifdef ROBOT
#include <blackdrops/safe_speed_control.hpp>
#endif

#include <blackdrops/blackdrops.hpp>
#include <blackdrops/gp_model.hpp>
#include <blackdrops/gp_multi_model.hpp>
#include <blackdrops/kernel_lf_opt.hpp>
#include <blackdrops/multi_gp.hpp>
#include <blackdrops/multi_gp_whole_opt.hpp>
#include <blackdrops/parallel_gp.hpp>
#include <limbo/experimental/model/poegp.hpp>
#include <limbo/experimental/model/poegp/poegp_lf_opt.hpp>

#include <blackdrops/nn_policy.hpp>

template <typename T>
inline T gaussian_rand(T m = 0.0, T v = 1.0)
{
    static std::random_device rd;
    static std::mt19937 gen(rd());

    std::normal_distribution<T> gaussian(m, v);

    return gaussian(gen);
}

inline double angle_dist(double a, double b)
{
    double theta = b - a;
    while (theta < -M_PI)
        theta += 2 * M_PI;
    while (theta > M_PI)
        theta -= 2 * M_PI;
    return theta;
}

struct Params {
#ifdef GRAPHIC
    struct graphics : robot_dart::defaults::graphics {
    };
#endif
    BO_DYN_PARAM(size_t, parallel_evaluations);
#ifdef ROBOT
    BO_PARAM(double, min_height, 0.1);
#endif

    struct options {
        BO_PARAM(bool, bounded, true);
    };

    struct blackdrops {
        BO_PARAM(double, dt, 0.1);
        BO_PARAM(double, t, 4.0);
        BO_PARAM(size_t, action_dim, 5);
        BO_PARAM(size_t, state_full_dim, 10);
        BO_PARAM(size_t, model_input_dim, 10);
        BO_PARAM(size_t, model_pred_dim, 10);
        BO_PARAM(size_t, rollout_steps, std::round(t() / dt()));
        BO_DYN_PARAM(double, boundary);
        BO_DYN_PARAM(bool, verbose);
    };

    struct damages {
        BO_PARAM_ARRAY(double, vel_offsets, 0., 0., 0., 0., 0.);
        BO_PARAM_ARRAY(double, offsets, 0., 0., 0., 0., 0.);
        BO_PARAM_ARRAY(int, blocked_jnts, 2);
        BO_PARAM_ARRAY(double, blocked_values, -M_PI / 6.0);
    };

    struct gp_model {
        BO_PARAM(double, noise, 0.01);
    };

    struct model_poegp : public limbo::defaults::model_poegp {
        BO_PARAM(int, leaf_size, 100);
        BO_PARAM(double, tau, 0.05);
    };

    struct model_gpmm : public limbo::defaults::model_gpmm {
        BO_PARAM(int, threshold, 200);
    };

    struct mean_constant {
        BO_PARAM(double, constant, 0.0);
    };

    struct kernel : public limbo::defaults::kernel {
        BO_PARAM(double, noise, gp_model::noise());
        BO_PARAM(bool, optimize_noise, true);
    };

    struct kernel_squared_exp_ard : public limbo::defaults::kernel_squared_exp_ard {
    };

    struct opt_cmaes : public limbo::defaults::opt_cmaes {
        BO_DYN_PARAM(double, max_fun_evals);
        BO_DYN_PARAM(double, fun_tolerance);
        BO_DYN_PARAM(int, restarts);
        BO_DYN_PARAM(int, elitism);
        BO_DYN_PARAM(bool, handle_uncertainty);

        BO_PARAM(int, variant, aBIPOP_CMAES);
        BO_PARAM(bool, verbose, false);
        BO_PARAM(bool, fun_compute_initial, true);
        BO_DYN_PARAM(double, ubound);
        BO_DYN_PARAM(double, lbound);
    };

    struct opt_nloptnograd : public limbo::defaults::opt_nloptnograd {
        BO_PARAM(int, iterations, 100);
    };

    struct opt_rprop : public limbo::defaults::opt_rprop {
        BO_PARAM(int, iterations, 150);
        BO_PARAM(double, eps_stop, 1e-4);
    };

    struct opt_parallelrepeater : public limbo::defaults::opt_parallelrepeater {
        BO_PARAM(int, repeats, 3);
    };
};

struct PolicyParams {
    struct blackdrops : public Params::blackdrops {
    };

    struct nn_policy {
        BO_PARAM(size_t, state_dim, Params::blackdrops::model_input_dim());
        BO_PARAM(size_t, action_dim, Params::blackdrops::action_dim());
// Velocity limits
#ifndef ROBOT
        BO_PARAM_ARRAY(double, max_u, 3.0, 3.46, 1.67, 3.0, 3.11);
        BO_PARAM_ARRAY(double, limits, 3.0, 3.46, 1.67, 3.0, 3.11, M_PI, 1.8, 1.82, M_PI, M_PI / 2.0);
#else
        // BO_PARAM_ARRAY(double, max_u, 1.88, 3.56, 1.67, 1.88, 4.77); // real-ones
        // BO_PARAM_ARRAY(double, max_u, 1.5, 2.5, 1.5, 1.5, 3.0); // safer-ones
        BO_PARAM_ARRAY(double, max_u, 1., 1., 1., 1., 1.); // safer-ones
        BO_PARAM_ARRAY(double, limits, 1., 1., 1., 1., 1., M_PI, 1.8, 1.82, M_PI, M_PI / 2.0);
#endif
        BO_DYN_PARAM(int, hidden_neurons);
        BO_PARAM(double, af, 0.5);
    };
};

namespace global {
    // FK/IK solvers
    std::shared_ptr<TRAC_IK::TRAC_IK> _ik_solver;
    std::shared_ptr<KDL::ChainFkSolverPos_recursive> _fk_solver;

    std::shared_ptr<robot_dart::Robot> global_robot; //, door_robot;
    dart::dynamics::SkeletonPtr object_skel;
    Eigen::VectorXd init_goal, goal;
    std::vector<dart::dynamics::SkeletonPtr> available_robots;
    std::vector<bool> used_robots, recreating_robots;
    std::mutex robot_mutex;
    int n_robots = 100;
    bool end = false;

#ifdef ROBOT
    std::shared_ptr<dynamixel::SafeSpeedControl> robot_control;
#endif
}

#ifdef ROBOT
bool init_robot(const std::string& usb_port)
{
    std::map<dynamixel::SafeSpeedControl::id_t, double> min_angles = {{1, 0.0}, {2, 1.6}, {3, 1.65}, {4, 0.0}, {5, 1.6}};
    std::map<dynamixel::SafeSpeedControl::id_t, double> max_angles = {{1, 2 * M_PI}, {2, 4.62}, {3, 4.2}, {4, 2 * M_PI}, {5, 4.62}};
    // conservative velocity limits
    std::map<dynamixel::SafeSpeedControl::id_t, double> max_velocities = {{1, 1.88}, {2, 3.56}, {3, 1.67}, {4, 1.88}, {5, 4.77}};

    std::unordered_set<dynamixel::protocols::Protocol1::id_t> selected_servos = {1, 2, 3, 4, 5};

    std::map<dynamixel::SafeSpeedControl::id_t, double> offsets = {{1, -6. * M_PI / 8.0}, {2, -M_PI}, {3, -M_PI}, {4, -M_PI}, {5, -M_PI}};

    try {
        global::robot_control = std::make_shared<dynamixel::SafeSpeedControl>(usb_port, selected_servos, min_angles, max_angles, max_velocities, Params::min_height(), offsets);
    }
    catch (dynamixel::errors::Error e) {
        std::cerr << "Dynamixel error:\n\t" << e.msg() << std::endl;
        return false;
    }

    return true;
}

void reset_robot()
{
    std::vector<double> target(5, 0.0);
    // blocked joints
    for (size_t i = 0; i < Params::damages::blocked_jnts_size(); i++) {
        if (Params::damages::blocked_jnts(i) == -1)
            continue;
        target[Params::damages::blocked_jnts(i)] = Params::damages::blocked_values(i);
    }

    bool reset = true;
    while (reset) {
        // move to initial position
        // global::robot_control->init_position();
        global::robot_control->go_to_target(target);
        std::cout << "Reset again? " << std::endl;
        // std::cin >> reset;
        sleep(2);
        reset = false;
    }
}
#endif

dart::dynamics::SkeletonPtr get_available_robot()
{
    std::lock_guard<std::mutex> lock(global::robot_mutex);
    for (int i = 0; i < global::n_robots; i++) {
        // std::lock_guard<std::mutex> lock(global::robot_mutex);
        if (global::used_robots[i] == false && global::recreating_robots[i] == false) {
            // std::lock_guard<std::mutex> lock(global::robot_mutex);
            // std::cout << "Giving robot '" << i << "'" << std::endl;
            global::used_robots[i] = true;
            return global::available_robots[i];
        }
    }

    return nullptr;
}

bool release_robot(const dart::dynamics::SkeletonPtr& robot)
{
    std::lock_guard<std::mutex> lock(global::robot_mutex);
    for (int i = 0; i < global::n_robots; i++) {
        // std::lock_guard<std::mutex> lock(global::robot_mutex);
        if (global::available_robots[i] == robot) {
            // std::lock_guard<std::mutex> lock(global::robot_mutex);
            // std::cout << "Releasing robot '" << i << "'" << std::endl;
            global::used_robots[i] = false;
            // global::recreating_robots[i] = true;
            global::available_robots[i]->clearExternalForces();
            global::available_robots[i]->clearInternalForces();
            global::available_robots[i]->clearConstraintImpulses();
            global::available_robots[i]->resetPositions();
            global::available_robots[i]->resetVelocities();
            global::available_robots[i]->resetAccelerations();
            global::available_robots[i]->resetGeneralizedForces();
            global::available_robots[i]->resetCommands();
            return true;
        }
    }

    return false;
}

void recreate_robots()
{
    while (!global::end) {
        std::lock_guard<std::mutex> lock(global::robot_mutex);
        for (int i = 0; i < global::n_robots; i++) {
            // std::lock_guard<std::mutex> lock(global::robot_mutex);
            if (global::recreating_robots[i]) {
                // std::lock_guard<std::mutex> lock(global::robot_mutex);
                // std::cout << "Recreating robot '" << i << "'" << std::endl;
                global::available_robots[i]->clearExternalForces();
                global::available_robots[i]->clearInternalForces();
                global::available_robots[i]->clearConstraintImpulses();
                global::available_robots[i]->resetPositions();
                global::available_robots[i]->resetVelocities();
                global::available_robots[i]->resetAccelerations();
                global::available_robots[i]->resetGeneralizedForces();
                global::available_robots[i]->resetCommands();
                for (int j = 0; j < global::available_robots[i]->getNumDofs(); j++) {
                    auto d = global::available_robots[i]->getDof(j);
                    d->resetPosition();
                    d->resetVelocity();
                    d->resetAcceleration();
                    d->resetForce();
                    d->resetVelocityChange();
                    d->resetConstraintImpulse();
                    d->resetCommand();
                }
                for (int j = 0; j < global::available_robots[i]->getNumJoints(); j++) {
                    auto d = global::available_robots[i]->getJoint(j);
                    d->resetPositions();
                    d->resetVelocities();
                    d->resetAccelerations();
                    d->resetForces();
                    d->resetVelocityChanges();
                    d->resetConstraintImpulses();
                    d->resetCommands();
                }
                global::recreating_robots[i] = false;
            }
        }
    }
}

Eigen::VectorXd get_robot_state(const std::shared_ptr<robot_dart::Robot>& robot) //, const std::shared_ptr<robot_dart::Robot>& ball)
{
    Eigen::VectorXd pos = robot->skeleton()->getPositions();
    Eigen::VectorXd vels = robot->skeleton()->getVelocities();
    size_t size = vels.size() + pos.size(); // + 3;
    Eigen::VectorXd state(size);
    state.head(vels.size()) = vels;
    state.segment(vels.size(), pos.size()) = pos;
    // state.tail(1) = ball->skeleton()->getPositions();

    return state;
}

class PolicyControl : public robot_dart::RobotControl {
public:
    using robot_t = std::shared_ptr<robot_dart::Robot>;

    PolicyControl() {}
    PolicyControl(const std::vector<double>& ctrl, robot_t robot)
        : robot_dart::RobotControl(ctrl, robot)
    {
        // _robot->set_actuator_types(dart::dynamics::Joint::SERVO);
        size_t _start_dof = 0;
        // if (!_robot->fixed_to_world()) {
        //     _start_dof = 6;
        // }
        std::vector<size_t> indices;
        std::vector<dart::dynamics::Joint::ActuatorType> types;
        for (size_t i = _start_dof; i < _dof; i++) {
            auto j = _robot->skeleton()->getDof(i)->getJoint();
            indices.push_back(_robot->skeleton()->getIndexOf(j));
            types.push_back(dart::dynamics::Joint::SERVO);
        }
        _robot->set_actuator_types(indices, types);

        _prev_time = 0.0;
        _t = 0.0;
        // _policy.set_params(Eigen::VectorXd::Map(ctrl.data(), ctrl.size()));
    }

    void update(double t)
    {
        _t = t;
        set_commands();
    }

    void set_commands()
    {
        double dt = Params::blackdrops::dt();

        // std::cout << _t << " vs " << _prev_time << " --> " << (_t - _prev_time) << std::endl;
        if (_t == 0.0 || (_t - _prev_time - dt) >= -1e-5) {
            // std::cout << "in" << std::endl;
            // std::cout << _t << " vs " << _prev_time << " --> " << (_t - _prev_time) << std::endl;
            Eigen::VectorXd state = get_robot_state(_robot); //, door);
            Eigen::VectorXd vel = state.head(5);
            Eigen::VectorXd pos = state.segment(5, 5);
            // // apply offsets/sensor damages
            // for (size_t i = 0; i < 5; i++) {
            //     pos(i) += Params::damages::offsets(i);
            // }
            // state.segment(5, 5) = pos;
            Eigen::VectorXd commands = policy.next(state);

            qs->push_back(pos);
            vels->push_back(vel);
            coms->push_back(commands);
            // doors->push_back(state.tail(1)[0]);

            // blocked joints
            for (size_t i = 0; i < Params::damages::blocked_jnts_size(); i++) {
                if (Params::damages::blocked_jnts(i) == -1)
                    continue;
                commands(Params::damages::blocked_jnts(i)) = 0.0;
                _robot->skeleton()->setPosition(Params::damages::blocked_jnts(i), Params::damages::blocked_values(i));
            }

            assert(_dof == (size_t)commands.size());
            // apply velocity offsets
            for (size_t i = 0; i < 5; i++) {
                commands(i) += Params::damages::vel_offsets(i);
            }
            _robot->skeleton()->setCommands(commands);
            _prev_commands = commands;
            _prev_time = _t; //int(_t / dt) * dt;
        }
        else
            _robot->skeleton()->setCommands(_prev_commands);
    }

    std::vector<Eigen::VectorXd>* qs;
    std::vector<Eigen::VectorXd>* vels;
    std::vector<Eigen::VectorXd>* coms;
    // std::vector<double>* doors;
    // std::shared_ptr<robot_dart::Robot> door;
    blackdrops::NNPolicy<PolicyParams> policy;

protected:
    double _prev_time;
    double _t;
    Eigen::VectorXd _prev_commands;
};

struct Omnigrasper {
    template <typename Policy, typename Reward>
    std::vector<std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd>> execute(const Policy& policy, const Reward& world, size_t steps, std::vector<double>& R, bool display = true)
    {
        double t = Params::blackdrops::t();
#ifndef ROBOT
#ifndef GRAPHIC
        using robot_simu_t = robot_dart::RobotDARTSimu<robot_dart::robot_control<PolicyControl>>; //, robot_dart::collision<dart::collision::FCLCollisionDetector>>;
#else
        using robot_simu_t = robot_dart::RobotDARTSimu<robot_dart::robot_control<PolicyControl>, robot_dart::graphics<robot_dart::Graphics<Params>>>; //, robot_dart::collision<dart::collision::FCLCollisionDetector>, robot_dart::graphics<robot_dart::Graphics<Params>>>;
#endif

        Eigen::VectorXd ctrl_params = policy.params();

        std::vector<double> ctrl(ctrl_params.size(), 0.0);
        Eigen::VectorXd::Map(ctrl.data(), ctrl.size()) = ctrl_params;

        auto c_robot = global::global_robot->clone();
        // c_robot->fix_to_world();
        // c_robot->set_position_enforced(true);

        // auto c_door = global::door_robot->clone();

        auto simu = robot_simu_t(ctrl, c_robot);
        simu.controller().policy = policy;
        // simu.controller().door = c_door;

        // Eigen::Vector6d pose = Eigen::VectorXd::Zero(6);
        // pose.tail(3) = Eigen::Vector3d(-0.6, 0.0, 0.0);
        // pose.head(3) = Eigen::Vector3d(0, 0, -dart::math::constants<double>::pi() / 2.0);
        // simu.add_skeleton(c_door->skeleton(), pose, "fixed", "door");

        simu.set_step(0.01);
        simu.add_floor();

        std::vector<Eigen::VectorXd> qs, coms, vels;
        // std::vector<double> doors;

        simu.controller().qs = &qs;
        simu.controller().vels = &vels;
        simu.controller().coms = &coms;
        // simu.controller().doors = &doors;

        // #ifdef GRAPHIC
        //         simu.graphics()->fixed_camera(Eigen::Vector3d(1.0, 1.0, 1.5), Eigen::Vector3d(-0.6, 0.0, 0.0));
        // #endif

        simu.run(t);

        // assert(doors.size() == 101);
        R = std::vector<double>();

        std::vector<std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd>> ret;

        for (size_t i = 0; i < vels.size() - 1; i++) {
            Eigen::VectorXd state(Params::blackdrops::model_input_dim());
            state.head(5) = vels[i];
            state.segment(5, 5) = qs[i];
            // state.tail(1) = limbo::tools::make_vector(doors[i]);

            Eigen::VectorXd command = coms[i];

            Eigen::VectorXd to_state(Params::blackdrops::model_input_dim());
            to_state.head(5) = vels[i + 1];
            to_state.segment(5, 5) = qs[i + 1];
            // to_state.tail(1) = limbo::tools::make_vector(doors[i + 1]);

            ret.push_back(std::make_tuple(state, command, to_state - state));
            R.push_back(world(state, command, to_state));
        }
#else
        double dt = Params::blackdrops::dt();
        // Recording data
        std::vector<Eigen::VectorXd> vels, q, coms;

        // map for velocities --- defaults to zero
        std::map<dynamixel::protocols::Protocol2::id_t, double> velocities;
        velocities[1] = 0.0;
        velocities[2] = 0.0;
        velocities[3] = 0.0;
        velocities[4] = 0.0;
        velocities[5] = 0.0;

        bool reset_fail = true;
        while (reset_fail) {
            try {
                // reset robot
                reset_robot();
                reset_fail = false;
            }
            catch (dynamixel::errors::Error e) {
                reset_fail = true;
                std::cerr << "Dynamixel error:\n\t" << e.msg() << std::endl;
                std::cout << "Did you reset the power? Just press any key..." << std::endl;
                char c;
                std::cin >> c;
                init_robot("/dev/ttyUSB0");
            }
        }
        std::vector<double> timings;
        // used for timing
        auto prev_time = std::chrono::steady_clock::now();
        auto start_time = prev_time;
        bool initial = true;
        std::chrono::duration<double> total_elapsed = std::chrono::steady_clock::now() - start_time;
        do {
            std::chrono::duration<double> elapsed_time = std::chrono::steady_clock::now() - prev_time;
            double time_elapsed = elapsed_time.count();
            if (initial || (time_elapsed - dt) > -1e-6) {
                initial = false;
                // Update time
                prev_time = std::chrono::steady_clock::now();
                // statistics
                // timings.push_back(time_elapsed);
                // read latest joint values (angular position and speed)
                auto actuators_pos = global::robot_control->joint_angles_with_offsets();
                // for (size_t i = 0; i < 5; i++)
                //     actuators_pos[i] = actuators_pos[i] - M_PI;

                auto actuators_vels = global::robot_control->joint_vels();

                // // TO-DO: Check with Dorian what's wrong
                // if (std::abs(actuators_vels[4]) > 30) {
                //     actuators_vels[4] = 0.0;
                // }
                Eigen::VectorXd full_state = Eigen::VectorXd::Zero(actuators_pos.size() + actuators_vels.size());
                full_state.head(actuators_vels.size()) = Eigen::VectorXd::Map(actuators_vels.data(), actuators_vels.size());
                full_state.segment(actuators_vels.size(), actuators_pos.size()) = Eigen::VectorXd::Map(actuators_pos.data(), actuators_pos.size());

                // // convert to Eigen vectors
                // Eigen::VectorXd full_state = get_robot_state(actuators_state, true);
                // // std::cout << "f: " << full_state.transpose() << std::endl;
                // Eigen::VectorXd state = get_robot_state(actuators_state);

                // Query policy for next commands
                Eigen::VectorXd commands = policy.next(full_state);
                for (int i = 0; i < commands.size(); i++) {
                    velocities[i + 1] = commands(i);
                }

                // blocked joints
                for (size_t i = 0; i < Params::damages::blocked_jnts_size(); i++) {
                    if (Params::damages::blocked_jnts(i) == -1)
                        continue;
                    velocities[Params::damages::blocked_jnts(i) + 1] = 0.0;
                }

                // std::cout << "st: " << state.transpose() << " com: " << commands.transpose() << std::endl;

                // Update statistics
                vels.push_back(full_state.head(actuators_vels.size()));
                q.push_back(full_state.segment(actuators_vels.size(), actuators_pos.size()));
                coms.push_back(commands);
            }
            // Send commands
            global::robot_control->velocity_command(velocities);

            total_elapsed = std::chrono::steady_clock::now() - start_time;
        } while (total_elapsed.count() <= t);
        total_elapsed = std::chrono::steady_clock::now() - start_time;

        // global::robot_control->go_to_target({M_PI / 2.0, M_PI / 4.0, M_PI / 4.0, -M_PI / 4.0, -M_PI / 4.0});
        // auto actuators_pos = global::robot_control->joint_angles_with_offsets();
        // Eigen::VectorXd tmp = Eigen::VectorXd::Map(actuators_pos.data(), actuators_pos.size());
        // q.push_back(tmp);

        velocities[1] = 0.0;
        velocities[2] = 0.0;
        velocities[3] = 0.0;
        velocities[4] = 0.0;
        velocities[5] = 0.0;
        global::robot_control->velocity_command(velocities);

        std::vector<std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd>> ret;
        R = std::vector<double>();

        std::cout << "#: " << q.size() << " --> " << total_elapsed.count() << std::endl;
        // std::cout << "init state: " << q[0].transpose() << std::endl;
        for (size_t i = 1; i < q.size(); i++) {
            Eigen::VectorXd state(Params::blackdrops::state_full_dim());
            Eigen::VectorXd to_state(Params::blackdrops::state_full_dim());

            state.head(vels[i - 1].size()) = vels[i - 1];
            state.segment(vels[i - 1].size(), q[i - 1].size()) = q[i - 1];

            to_state.head(vels[i].size()) = vels[i];
            to_state.segment(vels[i].size(), q[i].size()) = q[i];

            ret.push_back(std::make_tuple(state, coms[i - 1], to_state - state));
            R.push_back(world(state, coms[i - 1], to_state));
        }
#endif

        if (!policy.random() && display) {
            double rr = std::accumulate(R.begin(), R.end(), 0.0);
            std::cout << "Reward: " << rr << std::endl;
        }

        return ret;
    }

    template <typename Policy, typename Model, typename Reward>
    void execute_dummy(const Policy& policy, const Model& model, const Reward& world, size_t steps, std::vector<double>& R, bool display = true) const
    {
        R = std::vector<double>();
        // init state
        Eigen::VectorXd init = Eigen::VectorXd::Zero(Params::blackdrops::model_pred_dim());
        // blocked joints
        for (size_t i = 0; i < Params::damages::blocked_jnts_size(); i++) {
            if (Params::damages::blocked_jnts(i) == -1)
                continue;
            init(5 + Params::damages::blocked_jnts(i)) = Params::damages::blocked_values(i);
        }
        for (size_t j = 0; j < steps; j++) {
            Eigen::VectorXd query_vec(Params::blackdrops::model_input_dim() + Params::blackdrops::action_dim());

            Eigen::VectorXd u = policy.next(init);
            query_vec.head(Params::blackdrops::model_input_dim()) = init;
            query_vec.tail(Params::blackdrops::action_dim()) = u;

            Eigen::VectorXd mu;
            Eigen::VectorXd sigma;
            std::tie(mu, sigma) = model.predictm(query_vec);

            Eigen::VectorXd final = init + mu;
            std::cerr << init.transpose() << " with " << u.transpose() << " ---> " << final.transpose() << std::endl;

            double r = world(init, u, final);
            R.push_back(r);
            init = final;
        }
    }

    template <typename Policy, typename Model, typename Reward>
    double predict_policy(const Policy& policy, const Model& model, const Reward& world, size_t steps) const
    {
        double reward = 0.0;
        // init state
        Eigen::VectorXd init = Eigen::VectorXd::Zero(Params::blackdrops::model_pred_dim());
        // blocked joints
        for (size_t i = 0; i < Params::damages::blocked_jnts_size(); i++) {
            if (Params::damages::blocked_jnts(i) == -1)
                continue;
            init(5 + Params::damages::blocked_jnts(i)) = Params::damages::blocked_values(i);
        }
        for (size_t j = 0; j < steps; j++) {
            // if (init.norm() > 50)
            //     break;
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

            Eigen::VectorXd final = init + mu;

            reward += world(init, u, final);
            init = final;
        }

        return reward;
    }
};

struct RewardFunction {
    double operator()(const Eigen::VectorXd& from_state, const Eigen::VectorXd& action, const Eigen::VectorXd& to_state) const
    {
        // Eigen::VectorXd goal_state(5);
        // goal_state << 0, 0, M_PI / 4.0, M_PI / 4.0, M_PI / 4.0;
        double s_c_sq = 0.1 * 0.1;

        // get goal position
        KDL::Frame end_effector_pose;
        KDL::JntArray q(5);
        for (uint j = 0; j < 5; j++) {
            q(j) = to_state(5 + j);
        }

        global::_fk_solver->JntToCart(q, end_effector_pose);
        Eigen::VectorXd end_eff = Eigen::VectorXd::Map(end_effector_pose.p.data, 3);
        double da = (global::goal - end_eff).squaredNorm();

        return std::exp(-0.5 / s_c_sq * da);
    }
};

#ifdef MEAN
class VelocityControl : public robot_dart::RobotControl {
public:
    using robot_t = std::shared_ptr<robot_dart::Robot>;

    VelocityControl() {}
    VelocityControl(const std::vector<double>& ctrl, robot_t robot)
        : robot_dart::RobotControl(ctrl, robot)
    {
        // _robot->set_actuator_types(dart::dynamics::Joint::SERVO);
        size_t _start_dof = 0;
        // if (!_robot->fixed_to_world()) {
        //     _start_dof = 6;
        // }
        std::vector<size_t> indices;
        std::vector<dart::dynamics::Joint::ActuatorType> types;
        for (size_t i = _start_dof; i < _dof; i++) {
            auto j = _robot->skeleton()->getDof(i)->getJoint();
            indices.push_back(_robot->skeleton()->getIndexOf(j));
            types.push_back(dart::dynamics::Joint::SERVO);
        }
        _robot->set_actuator_types(indices, types);

        _velocities = Eigen::VectorXd::Map(ctrl.data(), ctrl.size());
        _t = 0.0;
        _prev_time = 0.0;
    }

    void update(double t)
    {
        _t = t;
        set_commands();
    }

    void set_commands()
    {
        double dt = Params::blackdrops::dt();
        if (_t == 0.0 || (_t - _prev_time - dt) >= -1e-5) {
            _state = get_robot_state(_robot);

            _prev_time = _t;
        }

        assert(_dof == (size_t)_velocities.size());
        _robot->skeleton()->setCommands(_velocities);
    }

    Eigen::VectorXd _state;

protected:
    Eigen::VectorXd _velocities;
    double _t, _prev_time;
};

struct MeanFunc {

    MeanFunc(int dim_out = 1)
    {
        _dampings = Eigen::VectorXd(5);
        _dampings << 0.25, 4.47, 2.0, 0.25, 0.12;
        _frictions = Eigen::VectorXd(5);
        _frictions << 0.183333333, 3.278, 1.02666666667, 0.183333333, 0.183333333;

        _params = Eigen::VectorXd::Constant(10, 0.5);
        // _params << 0.5, 0.5, 0.5, 0.5, 0.5;
        // _params(0) = (0.2 + 1.0) / 2.0; // 0.2 offset
    }

    template <typename GP>
    Eigen::VectorXd operator()(const Eigen::VectorXd& v, const GP& gp) const
    {
        double dt = Params::blackdrops::dt();

        // Eigen::VectorXd state = v.head(10);
        // Eigen::VectorXd u = v.tail(5);
        //
        // Eigen::VectorXd to_state = state;
        // to_state.head(5) = (1.0 - _params.array()) * u.array() + _params.array() * state.head(5).array();
        // to_state.tail(5) = state.tail(5).array() + (1.0 - _params.array()) * u.array() * dt + _params.array() * state.head(5).array() * dt;
        //
        // if (_lower_limits.size()) {
        //     for (int i = 0; i < 5; i++) {
        //         if (to_state(5 + i) < _lower_limits(i))
        //             to_state(5 + i) = _lower_limits(i);
        //         if (to_state(5 + i) > _upper_limits(i))
        //             to_state(5 + i) = _upper_limits(i);
        //     }
        // }
        //
        // return (to_state - state);

        using robot_simu_t = robot_dart::RobotDARTSimu<robot_dart::robot_control<VelocityControl>>;
        // using robot_simu_t = robot_dart::RobotDARTSimu<robot_dart::robot_control<VelocityControl>, robot_dart::graphics<robot_dart::Graphics<Params>>>;

        Eigen::VectorXd ctrl_params = v.tail(5);
        Eigen::VectorXd params = _params;
        params.head(5) = _params.head(5).array() * _dampings.array() * 2.0;
        params.tail(5) = _params.tail(5).array() * _frictions.array() * 2.0;
        // for (int i = 0; i < _params.size(); i++)
        //     ctrl_params(i) += _params(i) * 2.0 - 1.0;
        // std::cout << v.tail(5).transpose() << " vs " << ctrl_params.transpose() << std::endl;

        std::vector<double> ctrl(ctrl_params.size(), 0.0);
        Eigen::VectorXd::Map(ctrl.data(), ctrl.size()) = ctrl_params;

        auto rob = get_available_robot();
        while (rob == nullptr)
            rob = get_available_robot();
        for (size_t i = 0; i < rob->getNumDofs(); i++) {
            auto j = rob->getDof(i)->getJoint();
            // j->setSpringStiffness(0, 0.0);
            j->setDampingCoefficient(0, params(i));
            j->setCoulombFriction(0, params(5 + i));
        }
        auto c_robot = std::make_shared<robot_dart::Robot>(rob);
        // auto c_robot = global::global_robot->clone();
        // c_robot->fix_to_world();
        // c_robot->set_position_enforced(true);

        auto simu = robot_simu_t(ctrl, c_robot);

        simu.set_step(0.01);
        simu.add_floor();

        Eigen::VectorXd velocities = v.head(5);
        Eigen::VectorXd positions = v.segment(5, 5);
        c_robot->skeleton()->setVelocities(velocities);
        c_robot->skeleton()->setPositions(positions);

        simu.run(dt + 0.01);
        // std::cout << simu.world()->getTime() << std::endl;

        // assert(doors.size() == 1);

        Eigen::VectorXd state = v.head(10);

        Eigen::VectorXd to_state = simu.controller()._state; //get_robot_state(c_robot);

        release_robot(c_robot->skeleton());

        return (to_state - state);
    }

    Eigen::VectorXd h_params() const { return _params; }

    void set_h_params(const Eigen::VectorXd& params)
    {
        _params = params;
    }

protected:
    Eigen::VectorXd _params, _dampings, _frictions;
};
#endif

void init_simu(const std::string& robot_file)
{
    global::global_robot = std::make_shared<robot_dart::Robot>(robot_dart::Robot(robot_file, {}, "arm", true));
    // global::door_robot = std::make_shared<robot_dart::Robot>(robot_dart::Robot(door_file, {}, "door", true));

    // Initialize FK/IK solvers
    global::_ik_solver = std::make_shared<TRAC_IK::TRAC_IK>(robot_file, "base_link", "end_eff", 0.005, 1e-5);
    KDL::Chain chain;
    KDL::JntArray ll, ul;

    bool valid = global::_ik_solver->getKDLChain(chain);

    if (!valid) {
        std::cerr << "There was no valid KDL chain found" << std::endl;
        return;
    }

    valid = global::_ik_solver->getKDLLimits(ll, ul);

    if (!valid) {
        std::cerr << "There were no valid KDL joint limits found" << std::endl;
        return;
    }

    assert(chain.getNrOfJoints() == ll.data.size());
    assert(chain.getNrOfJoints() == ul.data.size());

    global::_fk_solver = std::make_shared<KDL::ChainFkSolverPos_recursive>(chain);

    // get goal position
    KDL::Frame end_effector_pose;
    Eigen::VectorXd q_desired(5);
    // q_desired << 2., 0., 0.45, -0.4, 0.1;
    q_desired << M_PI / 6.0, 0, M_PI / 4.0, M_PI / 4.0, M_PI / 4.0;
    KDL::JntArray q(chain.getNrOfJoints());
    for (uint j = 0; j < ll.data.size(); j++) {
        q(j) = q_desired(j);
    }

    global::_fk_solver->JntToCart(q, end_effector_pose);
    global::goal = Eigen::VectorXd::Map(end_effector_pose.p.data, 3);

    global::available_robots.clear();
    global::used_robots.clear();
    global::recreating_robots.clear();

    global::global_robot->fix_to_world();
    global::global_robot->set_position_enforced(true);

    for (int i = 0; i < global::n_robots; i++) {
        global::available_robots.push_back(global::global_robot->skeleton()->clone());
        // global::available_robots.back()->fix_to_world();
        // global::available_robots.back()->set_position_enforced(true);
        global::used_robots.push_back(false);
        global::recreating_robots.push_back(false);
    }
}

// limbo::opt::eval_t optimize_policy(const Eigen::VectorXd& params, bool eval_grad = false)
// {
//     RewardFunction world;
//     blackdrops::NNPolicy<PolicyParams> policy;
//
//     policy.set_params(params.array());
//
//     double r = 0.0;
//
//     double t = Params::blackdrops::t();
//
//     using robot_simu_t = robot_dart::RobotDARTSimu<robot_dart::robot_control<PolicyControl>>; //, robot_dart::collision<dart::collision::FCLCollisionDetector>>;
//
//     Eigen::VectorXd ctrl_params = policy.params();
//
//     std::vector<double> ctrl(ctrl_params.size(), 0.0);
//     Eigen::VectorXd::Map(ctrl.data(), ctrl.size()) = ctrl_params;
//
//     // std::cout << "I want a robot!" << std::endl;
//     auto rob = get_available_robot();
//     while (rob == nullptr)
//         rob = get_available_robot();
//     auto c_robot = std::make_shared<robot_dart::Robot>(rob); // global::global_robot->clone();
//     c_robot->fix_to_world();
//     c_robot->set_position_enforced(true);
//
//     // auto c_door = global::door_robot->clone();
//
//     auto simu = robot_simu_t(ctrl, c_robot);
//     simu.controller().policy = policy;
//
//     simu.set_step(0.01);
//     simu.add_floor();
//
//     std::vector<Eigen::VectorXd> qs, coms, vels;
//     // std::vector<double> doors;
//
//     simu.controller().qs = &qs;
//     simu.controller().vels = &vels;
//     simu.controller().coms = &coms;
//
//     simu.run(t);
//
//     for (size_t i = 0; i < vels.size() - 1; i++) {
//         Eigen::VectorXd state(Params::blackdrops::model_input_dim());
//         state.head(5) = vels[i];
//         state.segment(5, 5) = qs[i];
//         // state.tail(1) = limbo::tools::make_vector(doors[i]);
//
//         Eigen::VectorXd command = coms[i];
//
//         Eigen::VectorXd to_state(Params::blackdrops::model_input_dim());
//         to_state.head(5) = vels[i + 1];
//         to_state.segment(5, 5) = qs[i + 1];
//
//         r += world(state, command, to_state);
//     }
//
//     // std::cout << "Do not need the robot!" << std::endl;
//     release_robot(c_robot->skeleton());
//     return limbo::opt::no_grad(r);
// }

BO_DECLARE_DYN_PARAM(size_t, Params, parallel_evaluations);
BO_DECLARE_DYN_PARAM(int, PolicyParams::nn_policy, hidden_neurons);
BO_DECLARE_DYN_PARAM(double, Params::blackdrops, boundary);
BO_DECLARE_DYN_PARAM(bool, Params::blackdrops, verbose);

BO_DECLARE_DYN_PARAM(double, Params::opt_cmaes, max_fun_evals);
BO_DECLARE_DYN_PARAM(double, Params::opt_cmaes, fun_tolerance);
BO_DECLARE_DYN_PARAM(double, Params::opt_cmaes, lbound);
BO_DECLARE_DYN_PARAM(double, Params::opt_cmaes, ubound);
BO_DECLARE_DYN_PARAM(int, Params::opt_cmaes, restarts);
BO_DECLARE_DYN_PARAM(int, Params::opt_cmaes, elitism);
BO_DECLARE_DYN_PARAM(bool, Params::opt_cmaes, handle_uncertainty);

int main(int argc, char** argv)
{
    bool uncertainty = false;
    bool verbose = false;
    int threads = tbb::task_scheduler_init::automatic;
    std::string usb_port = "/dev/ttyUSB0"; //, policy_file = "";

    namespace po = boost::program_options;
    po::options_description desc("Command line arguments");
    // clang-format off
    desc.add_options()("help,h", "Prints this help message")
    ("parallel_evaluations,p", po::value<int>(), "Number of parallel monte carlo evaluations for policy reward estimation.")
    ("hidden_neurons,n", po::value<int>(), "Number of hidden neurons in NN policy.")
    ("boundary,b", po::value<double>(), "Boundary of the values during the optimization.")
    ("max_evals,m", po::value<int>(), "Max function evaluations to optimize the policy.")
    ("tolerance,t", po::value<double>(), "Maximum tolerance to continue optimizing the function.")
    ("restarts,r", po::value<int>(), "Max number of restarts to use during optimization.")
    ("elitism,e", po::value<int>(), "Elitism mode to use [0 to 3].")
    ("uncertainty,u", po::bool_switch(&uncertainty)->default_value(false), "Enable uncertainty handling.")
    ("threads,d", po::value<int>(), "Max number of threads used by TBB")
    ("verbose,v", po::bool_switch(&verbose)->default_value(false), "Enable verbose mode.")
    ("usb,s", po::value<std::string>(), "Set the USB port for the robot");
    // ("policy,f", po::value<std::string>(), "Policy to replay");
    // clang-format on

    try {
        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help")) {
            std::cout << desc << std::endl;
            return 0;
        }

        po::notify(vm);

        if (vm.count("parallel_evaluations")) {
            int c = vm["parallel_evaluations"].as<int>();
            if (c < 0)
                c = 0;
            Params::set_parallel_evaluations(c);
        }
        else {
            Params::set_parallel_evaluations(100);
        }
        if (vm.count("threads")) {
            threads = vm["threads"].as<int>();
        }
        if (vm.count("usb")) {
            usb_port = vm["usb"].as<std::string>();
        }
        // if (vm.count("policy")) {
        //     policy_file = vm["policy"].as<std::string>();
        // }
        if (vm.count("hidden_neurons")) {
            int c = vm["hidden_neurons"].as<int>();
            if (c < 1)
                c = 1;
            PolicyParams::nn_policy::set_hidden_neurons(c);
        }
        else {
            PolicyParams::nn_policy::set_hidden_neurons(5);
        }
        if (vm.count("boundary")) {
            double c = vm["boundary"].as<double>();
            if (c < 0)
                c = 0;
            Params::blackdrops::set_boundary(c);
            Params::opt_cmaes::set_lbound(-c);
            Params::opt_cmaes::set_ubound(c);
        }
        else {
            Params::blackdrops::set_boundary(0);
            Params::opt_cmaes::set_lbound(-6);
            Params::opt_cmaes::set_ubound(6);
        }

        // Cmaes parameters
        if (vm.count("max_evals")) {
            int c = vm["max_evals"].as<int>();
            Params::opt_cmaes::set_max_fun_evals(c);
        }
        else {
            Params::opt_cmaes::set_max_fun_evals(10000);
        }
        if (vm.count("tolerance")) {
            double c = vm["tolerance"].as<double>();
            if (c < 0.1)
                c = 0.1;
            Params::opt_cmaes::set_fun_tolerance(c);
        }
        else {
            Params::opt_cmaes::set_fun_tolerance(1);
        }
        if (vm.count("restarts")) {
            int c = vm["restarts"].as<int>();
            if (c < 1)
                c = 1;
            Params::opt_cmaes::set_restarts(c);
        }
        else {
            Params::opt_cmaes::set_restarts(3);
        }
        if (vm.count("elitism")) {
            int c = vm["elitism"].as<int>();
            if (c < 0 || c > 3)
                c = 0;
            Params::opt_cmaes::set_elitism(c);
        }
        else {
            Params::opt_cmaes::set_elitism(0);
        }
    }
    catch (po::error& e) {
        std::cerr << "[Exception caught while parsing command line arguments]: " << e.what() << std::endl;
        return 1;
    }

#ifdef USE_TBB
    static tbb::task_scheduler_init init(threads);
#endif

    Params::blackdrops::set_verbose(verbose);
    Params::opt_cmaes::set_handle_uncertainty(uncertainty);

    std::cout << std::endl;
    std::cout << "Cmaes parameters:" << std::endl;
    std::cout << "  max_fun_evals = " << Params::opt_cmaes::max_fun_evals() << std::endl;
    std::cout << "  fun_tolerance = " << Params::opt_cmaes::fun_tolerance() << std::endl;
    std::cout << "  restarts = " << Params::opt_cmaes::restarts() << std::endl;
    std::cout << "  elitism = " << Params::opt_cmaes::elitism() << std::endl;
    std::cout << "  handle_uncertainty = " << Params::opt_cmaes::handle_uncertainty() << std::endl;
    std::cout << "  boundary = " << Params::blackdrops::boundary() << std::endl;
    std::cout << "  tbb threads = " << threads << std::endl;
    std::cout << std::endl;

#ifdef ROBOT
    if (!init_robot(usb_port)) {
        std::cerr << "Could not connect to the robot! Exiting...";
        exit(1);
    }

    std::cout << "Connected!" << std::endl;
#endif

    // Load robot files
    const char* env_p = std::getenv("RESIBOTS_DIR");
    // initialization of the simulation and the simulated robot
    if (env_p) //if the environment variable exists
        init_simu(std::string(std::getenv("RESIBOTS_DIR")) + "/share/arm_models/URDF/omnigrasper_hook_real.urdf"); //, std::string(std::getenv("RESIBOTS_DIR")) + "/share/robot_models/URDF/door.urdf");
    else //if it does not exist, we might be running this on the cluster
        init_simu("/nfs/hal01/kchatzil/Workspaces/ResiBots/share/arm_models/URDF/omnigrasper_hook_real.urdf"); //, "/nfs/hal01/kchatzil/Workspaces/ResiBots/share/robot_models/URDF/door.urdf");

    using policy_opt_t = limbo::opt::Cmaes<Params>;

    using kernel_t = limbo::kernel::SquaredExpARD<Params>;
#ifndef MEAN
    using mean_t = limbo::mean::Constant<Params>;
#else
    using mean_t = MeanFunc;
#endif

#ifndef MODELIDENT
    using GP_t = blackdrops::ParallelGP<Params, limbo::model::GP, kernel_t, mean_t, limbo::model::gp::KernelLFOpt<Params>>; //, limbo::opt::NLOptGrad<Params, nlopt::LD_SLSQP>>>;
#else
    using GP_t = blackdrops::MultiGP<Params, limbo::model::GP, kernel_t, mean_t, blackdrops::MultiGPWholeLFOpt<Params, limbo::opt::NLOptNoGrad<Params, nlopt::LN_SBPLX>>>;
#endif

#ifdef SPGPS
#ifndef MODELIDENT
    using SPGP_t = blackdrops::ParallelGP<Params, limbo::experimental::model::POEGP, kernel_t, mean_t, limbo::experimental::model::poegp::POEKernelLFOpt<Params>>;
#else
    using SPGP_t = blackdrops::MultiGP<Params, limbo::experimental::model::POEGP, kernel_t, mean_t, blackdrops::MultiGPWholeLFOpt<Params, limbo::opt::NLOptNoGrad<Params, nlopt::LN_SBPLX>, limbo::experimental::model::poegp::POEKernelLFOpt<Params>>>;
#endif
    using GPMM_t = limbo::model::GPMultiModel<Params, GP_t, SPGP_t>;
    using MGP_t = blackdrops::GPModel<Params, GPMM_t>;
#else
    using MGP_t = blackdrops::GPModel<Params, GP_t>;
#endif

    // auto fut = std::async(std::launch::async, recreate_robots);
    // auto start = std::chrono::high_resolution_clock::now();
    blackdrops::BlackDROPS<Params, MGP_t, Omnigrasper, blackdrops::NNPolicy<PolicyParams>, policy_opt_t, RewardFunction> omni_reacher;

    omni_reacher.learn(1, 10, true);

    // // Let's just replay
    // if (policy_file == "") {
    //     std::cerr << "No file for replay provided.." << std::endl;
    //     exit(1);
    // }
    // if (policy_file == "target") {
    //     std::vector<double> target = {M_PI / 6.0, 0, M_PI / 4.0, M_PI / 4.0, M_PI / 4.0};
    //     global::robot_control->go_to_target(target);
    //     exit(1);
    // }
    // Omnigrasper robot;
    // blackdrops::NNPolicy<PolicyParams> policy;
    // policy.set_random_policy();
    // Eigen::VectorXd pp = policy.params();
    // Eigen::read_binary(policy_file, pp);
    // policy.set_params(pp);
    // std::vector<double> R;
    // RewardFunction world;
    // // Execute best policy so far on robot
    // auto obs_new = robot.execute(policy, world, Params::blackdrops::rollout_steps(), R);

    // std::vector<Eigen::VectorXd> states, commands, states_mean;
    // for (size_t i = 0; i < obs_new.size(); i++) {
    //     Eigen::VectorXd st, act, pred;
    //     st = std::get<0>(obs_new[i]);
    //     act = std::get<1>(obs_new[i]);
    //     pred = std::get<2>(obs_new[i]);

    //     states.push_back(st);
    //     commands.push_back(act);
    // }

    // MeanFunc model;

    // // init state
    // Eigen::VectorXd init_state = Eigen::VectorXd::Zero(Params::blackdrops::model_pred_dim());
    // for (size_t j = 0; j < states.size(); j++) {
    //     states_mean.push_back(init_state);
    //     Eigen::VectorXd query_vec(Params::blackdrops::model_input_dim() + Params::blackdrops::action_dim());

    //     Eigen::VectorXd u = commands[j];
    //     query_vec.head(Params::blackdrops::model_input_dim()) = init_state;
    //     query_vec.tail(Params::blackdrops::action_dim()) = u;

    //     Eigen::VectorXd mu = model(query_vec, query_vec);

    //     Eigen::VectorXd final = init_state + mu;

    //     // double r = world(init, mu, final);
    //     // R.push_back(r);
    //     init_state = final;
    // }

    // assert(states.size() == states_mean.size());
    // int n_errors = 0;
    // for (int i = 0; i < states.size(); i++) {
    //     std::cout << states[i].transpose() << " vs " << states_mean[i].transpose() << std::endl;
    //     if (states[i].isApprox(states_mean[i], 1e-5))
    //         std::cout << "Equal" << std::endl;
    //     else {
    //         std::cout << "Not equal!" << std::endl;
    //         n_errors++;
    //     }
    // }

    // std::cout << "# errors: " << n_errors << std::endl;

    // // auto time1 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count();
    // // std::cout << "Time(s): " << time1 * 1e-6 << std::endl;
    global::end = true;

    return 0;
}
