#include <limbo/experimental/model/spgp.hpp>
#include <limbo/limbo.hpp>
#include <limbo/mean/constant.hpp>

#include <medrops/safe_torque_control.hpp>

#include <robot_dart/robot_dart_simu.hpp>
#include <robot_dart/position_control.hpp>
#ifdef GRAPHIC
#include <robot_dart/graphics.hpp>
#endif

#include <boost/program_options.hpp>

#include <medrops/cmaes.hpp>
#include <medrops/exp_sq_ard.hpp>
#include <medrops/gp_model.hpp>
#include <medrops/gp_multi_model.hpp>
#include <medrops/gp_policy.hpp>
#include <medrops/kernel_lf_opt.hpp>
#include <medrops/linear_policy.hpp>
#include <medrops/medrops.hpp>
#include <medrops/sf_nn_policy.hpp>

template <typename T>
inline T gaussian_rand(T m = 0.0, T v = 1.0)
{
    static std::random_device rd;
    static std::mt19937 gen(rd());

    std::normal_distribution<T> gaussian(m, v);

    return gaussian(gen);
}

struct Params {
#ifdef GRAPHIC
    struct graphics : robot_dart::defaults::graphics {
    };
#endif
    BO_PARAM(size_t, action_dim, 3);
    BO_PARAM(size_t, state_full_dim, 12);
    BO_PARAM(size_t, model_input_dim, 9);
    BO_PARAM(size_t, model_pred_dim, 6);

    BO_PARAM(double, min_height, 0.0);

    BO_DYN_PARAM(size_t, parallel_evaluations);
    BO_DYN_PARAM(bool, verbose);

    struct options {
        BO_PARAM(bool, bounded, true);
    };

    struct medrops {
        BO_PARAM(size_t, rollout_steps, 36);
        BO_DYN_PARAM(double, boundary);
    };

    struct gp_model {
        BO_PARAM(double, noise, 1e-5);
    };
    struct model_spgp : public limbo::defaults::model_spgp {
        BO_PARAM(double, samples_percent, 10);
        BO_PARAM(double, jitter, 1e-5);
        BO_PARAM(int, min_m, 100);
        BO_PARAM(double, sig, 0.001);
    };
    struct model_gpmm : public limbo::defaults::model_gpmm {
        BO_PARAM(int, threshold, 300);
    };

    struct linear_policy {
        BO_PARAM(int, state_dim, 9);
        BO_PARAM_ARRAY(double, max_u, 0.5, 0.5, 0.5);
    };

    struct nn_policy {
        BO_PARAM(int, state_dim, 9);
        BO_PARAM_ARRAY(double, max_u, 0.7, 0.7, 0.7);
        BO_DYN_PARAM(int, hidden_neurons);
    };

    struct gp_policy {
        BO_PARAM_ARRAY(double, max_u, 0.5, 0.5, 0.5);
        BO_PARAM(double, pseudo_samples, 10);
        BO_PARAM(double, noise, 0.01);
        BO_PARAM(int, state_dim, 9);
    };

    struct mean_constant {
        BO_PARAM(double, constant, 0.0);
    };

    struct opt_nloptgrad : public limbo::defaults::opt_nloptgrad {
        BO_PARAM(int, iterations, 1000);
    };

    struct kernel_squared_exp_ard : public limbo::defaults::kernel_squared_exp_ard {
    };

    struct kernel_exp : public limbo::defaults::kernel_exp {
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
        // BO_PARAM(double, fun_target, 30);
        BO_DYN_PARAM(double, ubound);
        BO_DYN_PARAM(double, lbound);
        BO_DYN_PARAM(int, lambda);
    };

    struct opt_nloptnograd : public limbo::defaults::opt_nloptnograd {
        BO_PARAM(int, iterations, 20000);
    };
};

inline double angle_dist(double a, double b)
{
    double theta = b - a;
    while (theta < -M_PI)
        theta += 2 * M_PI;
    while (theta > M_PI)
        theta -= 2 * M_PI;
    return theta;
}

namespace data {
    std::vector<Eigen::VectorXd> vels, poses, coms;
}

namespace global {
    std::shared_ptr<robot_dart::Robot> global_robot;
#ifndef GPPOLICY
    using policy_t = medrops::SFNNPolicy<Params>;
#else
    using policy_t = medrops::GPPolicy<Params>;
#endif

    Eigen::VectorXd goal(3);

    using kernel_t = limbo::kernel::Exp<Params>; //medrops::SquaredExpARD<Params>; //limbo::kernel::Exp<Params>;
    using mean_t = limbo::mean::Constant<Params>;

    using GP_t = limbo::model::GP<Params, kernel_t, mean_t>; //, medrops::KernelLFOpt<Params, limbo::opt::NLOptGrad<Params, nlopt::LD_SLSQP>>>;
    GP_t reward_gp(3, 1);

    std::shared_ptr<dynamixel::SafeTorqueControl> robot_control;
}

bool init_robot(const std::string& usb_port)
{
    std::map<dynamixel::SafeTorqueControl::id_t, double> min_angles = {{1, 1.57}, {2, 2.09}, {3, 1.98}, {4, 1.57}};
    std::map<dynamixel::SafeTorqueControl::id_t, double> max_angles = {{1, 4.71}, {2, 4.19}, {3, 4.3}, {4, 4.3}};
    // conservative torque limits
    std::map<dynamixel::SafeTorqueControl::id_t, double> max_torques = {{1, 100}, {2, 120}, {3, 120}, {4, 100}};

    std::unordered_set<dynamixel::protocols::Protocol2::id_t> selected_servos = {1, 2, 3};

    try {
        global::robot_control = std::make_shared<dynamixel::SafeTorqueControl>(usb_port, selected_servos, min_angles, max_angles, max_torques, Params::min_height());
    }
    catch (dynamixel::errors::Error e) {
        std::cerr << "Dynamixel error:\n\t" << e.msg() << std::endl;
        return false;
    }

    return true;
}

void reset_robot()
{
    bool reset = true;
    while (reset) {
        // move to initial position
        global::robot_control->init_position();
        usleep(1.5 * 1e6);
        std::cout << "Reset again? " << std::endl;
        std::cin >> reset;
    }
    // move to torque control
    global::robot_control->set_torque_control_mode();
}

Eigen::VectorXd get_robot_state(const std::vector<double>& vec, bool full = false)
{
    size_t size = vec.size();
    if (full)
        size += 3;
    Eigen::VectorXd state(size);
    for (size_t i = 0; i < 3; i++)
        state(i) = vec[i];
    if (!full) {
        for (size_t i = 3; i < 6; i++)
            state(i) = vec[i];
    }
    else {
        for (int i = 0; i < 3; i++) {
            state(3 + 2 * i) = std::cos(vec[3 + i]);
            state(3 + 2 * i + 1) = std::sin(vec[3 + i]);
        }
    }
    return state;
}

struct ActualReward {
    double operator()(const Eigen::VectorXd& to_state) const
    {
        using robot_simu_t = robot_dart::RobotDARTSimu<robot_dart::robot_control<robot_dart::PositionControl>>;

        std::shared_ptr<robot_dart::Robot> simulated_robot = global::global_robot->clone();
        simulated_robot->fix_to_world();
        simulated_robot->set_position_enforced(true);

        std::vector<double> params(3, 1.0);
        for (int i = 0; i < to_state.size(); i++)
            params[i] = to_state(i);
        robot_simu_t simu(params, simulated_robot);
        simu.run(2);

        auto bd = simulated_robot->skeleton()->getBodyNode("arm_3_sub");
        Eigen::VectorXd eef = bd->getCOM();
        double s_c_sq = 0.1 * 0.1; //0.1 * 0.1; //, s_c_a = 0.1 * 0.1;
        double s_c = 0.25 * 0.25;
        double de = (eef - global::goal).squaredNorm();

        std::map<dynamixel::SafeTorqueControl::id_t, double> min_angles = {{1, 1.57}, {2, 2.09}, {3, 1.98}, {4, 1.57}};
        std::map<dynamixel::SafeTorqueControl::id_t, double> max_angles = {{1, 4.71}, {2, 4.19}, {3, 4.3}, {4, 4.3}};
        // double p1 = 0.0, p2 = 0.0;
        for (int i = 0; i < 3; i++) {
            double ll = min_angles[i + 1] - M_PI;
            double ul = max_angles[i + 1] - M_PI;
            if ((to_state(i) - ll) < 0.1) {
                return -10.0;
            }
            if ((ul - to_state(i)) < 0.1) {
                return -10.0;
            }
            // double el = to_state(i) + M_PI - min_angles[i + 1];
            // double eu = max_angles[i + 1] - to_state(i) - M_PI;
            // p1 -= std::exp(-0.5 / s_c_a * el * el);
            // p2 -= std::exp(-0.5 / s_c_a * eu * eu);
        }

        double z = 0.252 * std::cos(to_state(1) + to_state(2) + 2 * M_PI) - 0.264 * std::cos(to_state(1) + M_PI);
        if (z <= Params::min_height())
            return -10.0;

        return std::exp(-0.5 / s_c_sq * de); // + p1 + p2;
    }
};

struct Omnigrasper {
    template <typename Policy, typename Reward>
    std::vector<std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd>> execute(const Policy& policy, const Reward& world, size_t steps, std::vector<double>& R, bool display = true)
    {
        std::vector<std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd>> res;
        Eigen::VectorXd pp = policy.params();
        double t = 2.0, dt = 0.05;

        // Recording data
        std::vector<Eigen::VectorXd> vels, poses, coms;

        // map for torques --- defaults to zero
        std::map<dynamixel::protocols::Protocol2::id_t, double> torques;
        torques[1] = 0.0;
        torques[2] = 0.0;
        torques[3] = 0.0;

        bool limit_reached = false;
        bool movement_fail = true;

        while (movement_fail) {
            try {
                // reset robot
                reset_robot();
                // used for timing
                auto prev_time = std::chrono::steady_clock::now();
                auto start_time = prev_time;
                std::chrono::duration<double> total_elapsed = std::chrono::steady_clock::now() - start_time;
                do {
                    std::chrono::duration<double> elapsed_time = std::chrono::steady_clock::now() - prev_time;
                    if (elapsed_time.count() <= 1e-5 || elapsed_time.count() >= dt) {
                        // Update time
                        prev_time = std::chrono::steady_clock::now();
                        // read latest joint values (angular position and speed)
                        auto actuators_state = global::robot_control->concatenated_joint_state();
                        // Substract pi [physical joint angles are not centered around 0 but pi]
                        for (size_t i = 3; i < 6; i++)
                            actuators_state[i] = actuators_state[i] - M_PI;

                        // convert to Eigen vectors
                        Eigen::VectorXd full_state = get_robot_state(actuators_state, true);
                        Eigen::VectorXd state = get_robot_state(actuators_state);

                        // Query policy for next commands
                        Eigen::VectorXd commands = policy.next(full_state);
                        for (size_t i = 0; i < commands.size(); i++) {
                            torques[i + 1] = commands(i);
                        }

                        // std::cout << "st: " << state.transpose() << " com: " << commands.transpose() << std::endl;

                        // Update statistics
                        Eigen::VectorXd vel = state.head(3);
                        Eigen::VectorXd pos = state.tail(3);
                        vels.push_back(vel);
                        poses.push_back(pos);
                        coms.push_back(commands);

                        // Send commands
                        global::robot_control->torque_command(torques);
                    }
                    if (global::robot_control->enforce_joint_limits()) {
                        std::cout << "Reached joint limits!" << std::endl;
                        limit_reached = true;

                        auto actuators_state = global::robot_control->concatenated_joint_state();
                        // Substract pi [physical joint angles are not centered around 0 but pi]
                        for (size_t i = 3; i < 6; i++)
                            actuators_state[i] = actuators_state[i] - M_PI;

                        // convert to Eigen vectors
                        Eigen::VectorXd full_state = get_robot_state(actuators_state, true);
                        Eigen::VectorXd state = get_robot_state(actuators_state);
                        // Query policy for next commands
                        Eigen::VectorXd commands = policy.next(full_state);
                        for (size_t i = 0; i < commands.size(); i++) {
                            torques[i + 1] = commands(i);
                        }
                        // Update statistics
                        Eigen::VectorXd vel = state.head(3);
                        Eigen::VectorXd pos = state.tail(3);
                        vels.push_back(vel);
                        poses.push_back(pos);
                        coms.push_back(commands);

                        break;
                    }
                    total_elapsed = std::chrono::steady_clock::now() - start_time;
                    // std::cout << "Elasped: " << total_elapsed.count() << std::endl;
                } while (total_elapsed.count() <= t);
                total_elapsed = std::chrono::steady_clock::now() - start_time;
                movement_fail = false;
            }
            catch (dynamixel::errors::Error e) {
                movement_fail = true;
                std::cerr << "Dynamixel error:\n\t" << e.msg() << std::endl;
                std::cout << "Did you reset the power? Just press any key..." << std::endl;
                char c;
                std::cin >> c;
                init_robot("/dev/ttyUSB0");
            }
        }

        if (!limit_reached)
            global::robot_control->reset_to_position_control();

        R = std::vector<double>();

        ActualReward actual_reward;

        // if (display)
        std::cout << "#: " << vels.size() << std::endl;
        for (size_t j = 0; j < vels.size() - 1; j++) {
            size_t id = j; // * step;
            Eigen::VectorXd init(Params::model_pred_dim());
            init.head(3) = vels[id];
            // init.segment(3, 3) = poses[id];
            init.tail(3) = poses[id]; //qs[id];
            Eigen::VectorXd init_full(Params::model_input_dim());
            init_full.head(3) = init.head(3);
            for (int i = 0; i < 3; i++) {
                init_full(3 + 2 * i) = std::cos(init(3 + i));
                init_full(3 + 2 * i + 1) = std::sin(init(3 + i));
            }
            Eigen::VectorXd u = coms[id];
            Eigen::VectorXd final(Params::model_pred_dim());
            final.head(3) = vels[id + 1];
            final.tail(3) = poses[id + 1]; //qs[id + 1];

            // if (display) {
            // std::cout << "state: " << init.transpose() << std::endl;
            // std::cout << "command: " << u.transpose() << std::endl;
            // std::cout << poses[id].transpose() << " to " << poses[id + 1].transpose() << std::endl;
            // std::cout << "vel: " << vels[id].transpose() << std::endl;
            // std::cout << "vel: " << vels[id + 1].transpose() << std::endl;
            // std::cout << "my_vel: " << ((poses[id + 1] - poses[id]).array() / 0.05).transpose() << std::endl;
            // }
            // std::cout << "next state: " << final.transpose() << std::endl;
            double r = actual_reward(final.tail(3)); //world(init, u, final);
            // if (limit_reached && j == vels.size() - 2)
            //     r = -10;
            std::cout << final.transpose() << ": " << r << std::endl;
            std::cout << u.transpose() << std::endl;
            global::reward_gp.add_sample(final.tail(3), limbo::tools::make_vector(r), 0.001);
            R.push_back(r);
            res.push_back(std::make_tuple(init_full, u, final - init));
            if (r < 0)
                break;
        }

        //         std::shared_ptr<robot_dart::Robot> simulated_robot = global::global_robot->clone();
        //         simulated_robot->fix_to_world();
        //         simulated_robot->set_position_enforced(true);
        //
        // #ifdef GRAPHIC
        //         using robot_simu_t = robot_dart::RobotDARTSimu<robot_dart::robot_control<robot_dart::PositionControl>, robot_dart::graphics<robot_dart::Graphics<Params>>>;
        // #else
        //         using robot_simu_t = robot_dart::RobotDARTSimu<robot_dart::robot_control<robot_dart::PositionControl>>;
        // #endif
        //         // using robot_simu_t = robot_dart::RobotDARTSimu<robot_dart::robot_control<robot_dart::PositionControl>>;
        //
        //         std::vector<double> params(3, 0.0);
        //         for (size_t ii = 0; ii < 3; ii++)
        //             params[ii] = poses[vels.size() - 1][ii];
        //
        //         robot_simu_t simu(params, simulated_robot);
        //         simu.run(5);
        // global::reward_gp.recompute();
        // global::reward_gp.optimize_hyperparams();

        if (!policy.random() && display) {
            double rr = std::accumulate(R.begin(), R.end(), 0.0);
            std::cout << "Reward: " << rr << std::endl;
        }

        return res;
    }

    template <typename Policy, typename Model, typename Reward>
    void execute_dummy(const Policy& policy, const Model& model, const Reward& world, size_t steps, std::vector<double>& R, bool display = true) const
    {
        R = std::vector<double>();
        // init state
        Eigen::VectorXd init = Eigen::VectorXd::Zero(Params::model_pred_dim());
        // init(5) = 0.58;
        for (size_t j = 0; j < steps; j++) {
            Eigen::VectorXd query_vec(Params::model_input_dim() + Params::action_dim());
            Eigen::VectorXd init_full(Params::model_input_dim());
            init_full.head(3) = init.head(3);
            for (int i = 0; i < 3; i++) {
                init_full(3 + 2 * i) = std::cos(init(3 + i));
                init_full(3 + 2 * i + 1) = std::sin(init(3 + i));
            }
            // init.tail(Params::model_input_dim()) = init.tail(Params::model_input_dim()).unaryExpr([](double x) { return angle_dist(0,x); });
            Eigen::VectorXd u = policy.next(init_full);
            query_vec.head(Params::model_input_dim()) = init_full;
            query_vec.tail(Params::action_dim()) = u;

            Eigen::VectorXd mu;
            Eigen::VectorXd sigma;
            std::tie(mu, sigma) = model.predictm(query_vec);

            Eigen::VectorXd final = init + mu;
            // std::cout << init.transpose() << " to " << final.transpose() << " dx: " << mu.transpose() << std::endl;
            // std::cout << "state: " << init.transpose() << std::endl;
            // std::cout << "command: " << u.transpose() << std::endl;
            // final.tail(Params::model_input_dim()) = final.tail(Params::model_input_dim()).unaryExpr([](double x) { return angle_dist(0,x); });

            double r = world(init, mu, final);
            R.push_back(r);
            init = final;
        }
    }

    template <typename Policy, typename Model, typename Reward>
    double predict_policy(const Policy& policy, const Model& model, const Reward& world, size_t steps) const
    {
        size_t N = Params::parallel_evaluations();

        Eigen::VectorXd rews(N);
        tbb::parallel_for(size_t(0), N, size_t(1), [&](size_t i) {
            // std::chrono::steady_clock::time_point time_start = std::chrono::steady_clock::now();

            double reward = 0.0;
            // init state
            Eigen::VectorXd init = Eigen::VectorXd::Zero(Params::model_pred_dim());
            // init(5) = 0.58;
            for (size_t j = 0; j < steps; j++) {
                Eigen::VectorXd query_vec(Params::model_input_dim() + Params::action_dim());
                Eigen::VectorXd init_full(Params::model_input_dim());
                init_full.head(3) = init.head(3);
                for (int i = 0; i < 3; i++) {
                    init_full(3 + 2 * i) = std::cos(init(3 + i));
                    init_full(3 + 2 * i + 1) = std::sin(init(3 + i));
                }
                Eigen::VectorXd u = policy.next(init_full);
                query_vec.head(Params::model_input_dim()) = init_full;
                query_vec.tail(Params::action_dim()) = u;

                Eigen::VectorXd mu;
                Eigen::VectorXd sigma;
                std::tie(mu, sigma) = model.predictm(query_vec);

                if (Params::parallel_evaluations() > 1 || Params::opt_cmaes::handle_uncertainty()) {
                    sigma = sigma.array().sqrt();
                    for (int i = 0; i < mu.size(); i++) {
                        double s = gaussian_rand(mu(i), sigma(i));
                        mu(i) = std::max(mu(i) - sigma(i),
                            std::min(s, mu(i) + sigma(i)));
                    }
                }

                Eigen::VectorXd final = init + mu;
                // final.tail(Params::model_input_dim()) = final.tail(Params::model_input_dim()).unaryExpr([](double x) { return angle_dist(0,x); });

                reward += world(init, u, final);
                init = final;
            }
            rews(i) = reward;

            // double rollout_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - time_start).count();
            //std::cout << "Rollout finished, took " << rollout_ms << "ms" << std::endl;
        });

        double r = rews(0);
        if (Params::parallel_evaluations() > 1) {
#ifdef MEDIAN
            r = Eigen::percentile_v(rews, 25) + Eigen::percentile_v(rews, 50) + Eigen::percentile_v(rews, 75);
#else
            r = rews.mean();
#endif
        }

        return r;
    }
};

struct RewardFunction {
    double operator()(const Eigen::VectorXd& from_state, const Eigen::VectorXd& action, const Eigen::VectorXd& to_state) const
    {
        // double s_c_sq = 0.25 * 0.25;
        // // double de = (to_state.tail(3) - global::goal).squaredNorm();
        // double de = 0.0;
        // for (size_t i = 0; i < 3; i++) {
        //     double dx = angle_dist(to_state(3 + i), global::goal(i));
        //     de += dx * dx;
        // }
        //
        // std::map<dynamixel::SafeTorqueControl::id_t, double> min_angles = {{1, 1.57}, {2, 2.09}, {3, 1.98}, {4, 1.57}};
        // std::map<dynamixel::SafeTorqueControl::id_t, double> max_angles = {{1, 4.71}, {2, 4.19}, {3, 4.3}, {4, 4.3}};
        // // double p1 = 0.0, p2 = 0.0;
        // for (int i = 0; i < 3; i++) {
        //     double ll = min_angles[i + 1] - M_PI;
        //     double ul = max_angles[i + 1] - M_PI;
        //     if (to_state(3 + i) < ll) {
        //         return -10.0;
        //     }
        //     if (to_state(3 + i) > ul) {
        //         return -10.0;
        //     }
        // }
        //
        // return std::exp(-0.5 / s_c_sq * de);
        return global::reward_gp.mu(to_state.tail(3))[0];
    }
};

void init_simu(const std::string& robot_file)
{
    global::global_robot = std::make_shared<robot_dart::Robot>(robot_dart::Robot(robot_file, {}, "arm", true));

    // get goal position
    std::shared_ptr<robot_dart::Robot> simulated_robot = global::global_robot->clone();
    simulated_robot->fix_to_world();
    simulated_robot->set_position_enforced(true);

    using robot_simu_t = robot_dart::RobotDARTSimu<robot_dart::robot_control<robot_dart::PositionControl>>;

    std::vector<double> params(3, 0.5);
    // params[2] = -0.5;
    // params[0] = -2.5;
    // params[2] = 1.0;

    robot_simu_t simu(params, simulated_robot);
    simu.run(2);
    auto bd = simulated_robot->skeleton()->getBodyNode("arm_3_sub");
    global::goal = bd->getCOM();
    // global::goal << 0.3, 0.3, 0.3;
    // global::goal = get_robot_state(simulated_robot).segment(3, 3);
    std::cout << "Goal is: " << global::goal.transpose() << std::endl;
    // global::goal = Eigen::VectorXd(3);
    // global::goal << 1, 1, 1;
}

using kernel_t = medrops::SquaredExpARD<Params>;
using mean_t = limbo::mean::Constant<Params>;

using GP_t = limbo::model::GP<Params, kernel_t, mean_t, medrops::KernelLFOpt<Params, limbo::opt::NLOptGrad<Params, nlopt::LD_SLSQP>>>;
using SPGP_t = limbo::model::SPGP<Params, kernel_t, mean_t>;

BO_DECLARE_DYN_PARAM(size_t, Params, parallel_evaluations);
BO_DECLARE_DYN_PARAM(int, Params::nn_policy, hidden_neurons);
BO_DECLARE_DYN_PARAM(double, Params::medrops, boundary);
BO_DECLARE_DYN_PARAM(bool, Params, verbose);

BO_DECLARE_DYN_PARAM(double, Params::opt_cmaes, max_fun_evals);
BO_DECLARE_DYN_PARAM(double, Params::opt_cmaes, fun_tolerance);
BO_DECLARE_DYN_PARAM(double, Params::opt_cmaes, lbound);
BO_DECLARE_DYN_PARAM(double, Params::opt_cmaes, ubound);
BO_DECLARE_DYN_PARAM(int, Params::opt_cmaes, lambda);
BO_DECLARE_DYN_PARAM(int, Params::opt_cmaes, restarts);
BO_DECLARE_DYN_PARAM(int, Params::opt_cmaes, elitism);
BO_DECLARE_DYN_PARAM(bool, Params::opt_cmaes, handle_uncertainty);

int main(int argc, char** argv)
{
    bool uncertainty = false;
    bool verbose = false;
    int threads = tbb::task_scheduler_init::automatic;
    std::string usb_port = "/dev/ttyUSB0";

    namespace po = boost::program_options;
    po::options_description desc("Command line arguments");
    // clang-format off
    desc.add_options()("help,h", "Prints this help message")
    ("parallel_evaluations,p", po::value<int>(), "Number of parallel monte carlo evaluations for policy reward estimation.")
    ("hidden_neurons,n", po::value<int>(), "Number of hidden neurons in NN policy.")
    ("boundary,b", po::value<double>(), "Boundary of the values during the optimization.")
    ("lambda,l", po::value<int>(), "Initial population in CMA-ES (-1 to default)")
    ("max_evals,m", po::value<int>(), "Max function evaluations to optimize the policy.")
    ("tolerance,t", po::value<double>(), "Maximum tolerance to continue optimizing the function.")
    ("restarts,r", po::value<int>(), "Max number of restarts to use during optimization.")
    ("elitism,e", po::value<int>(), "Elitism mode to use [0 to 3].")
    ("uncertainty,u", po::bool_switch(&uncertainty)->default_value(false), "Enable uncertainty handling.")
    ("threads,d", po::value<int>(), "Max number of threads used by TBB")
    ("verbose,v", po::bool_switch(&verbose)->default_value(false), "Enable verbose mode.")
    ("usb,s", po::value<std::string>(), "Set the USB port for the robot");
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
        if (vm.count("usb")) {
            usb_port = vm["usb"].as<std::string>();
        }
        if (vm.count("threads")) {
            threads = vm["threads"].as<int>();
        }
        if (vm.count("hidden_neurons")) {
            int c = vm["hidden_neurons"].as<int>();
            if (c < 1)
                c = 1;
            Params::nn_policy::set_hidden_neurons(c);
        }
        else {
            Params::nn_policy::set_hidden_neurons(5);
        }
        if (vm.count("boundary")) {
            double c = vm["boundary"].as<double>();
            if (c < 0)
                c = 0;
            Params::medrops::set_boundary(c);
            Params::opt_cmaes::set_lbound(-c);
            Params::opt_cmaes::set_ubound(c);
        }
        else {
            Params::medrops::set_boundary(0);
            Params::opt_cmaes::set_lbound(-6);
            Params::opt_cmaes::set_ubound(6);
        }

        int lambda = -1;
        if (vm.count("lambda")) {
            lambda = vm["lambda"].as<int>();
        }
        Params::opt_cmaes::set_lambda(lambda);

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

    Params::set_verbose(verbose);
    Params::opt_cmaes::set_handle_uncertainty(uncertainty);

    std::cout << std::endl;
    std::cout << "Cmaes parameters:" << std::endl;
    std::cout << "  max_fun_evals = " << Params::opt_cmaes::max_fun_evals() << std::endl;
    std::cout << "  fun_tolerance = " << Params::opt_cmaes::fun_tolerance() << std::endl;
    std::cout << "  restarts = " << Params::opt_cmaes::restarts() << std::endl;
    std::cout << "  elitism = " << Params::opt_cmaes::elitism() << std::endl;
    std::cout << "  handle_uncertainty = " << Params::opt_cmaes::handle_uncertainty() << std::endl;
    std::cout << "  lambda (CMA-ES population) = " << Params::opt_cmaes::lambda() << std::endl;
    std::cout << "  boundary = " << Params::medrops::boundary() << std::endl;
    std::cout << "  tbb threads = " << threads << std::endl;
    std::cout << std::endl;

    if (!init_robot(usb_port)) {
        std::cerr << "Could not connect to the robot! Exiting...";
        exit(1);
    }

    const char* env_p = std::getenv("RESIBOTS_DIR");
    // initilisation of the simulation and the simulated robot
    if (env_p) //if the environment variable exists
        init_simu(std::string(std::getenv("RESIBOTS_DIR")) + "/share/arm_models/URDF/omnigrasper_3dof.urdf");
    else //if it does not exist, we might be running this on the cluster
        init_simu("/nfs/hal01/kchatzil/Workspaces/ResiBots/share/arm_models/URDF/omnigrasper_3dof.urdf");

    using policy_opt_t = limbo::opt::CustomCmaes<Params>;
//using policy_opt_t = limbo::opt::NLOptGrad<Params>;
#ifdef SPGPS
    using GPMM_t = limbo::model::GPMultiModel<Params, mean_t, GP_t, SPGP_t>;
    using MGP_t = medrops::GPModel<Params, GPMM_t>;
#else
    using MGP_t = medrops::GPModel<Params, GP_t>;
#endif

#ifndef GPPOLICY
    medrops::Medrops<Params, MGP_t, Omnigrasper, medrops::SFNNPolicy<Params>, policy_opt_t, RewardFunction> cp_system;
#else
    medrops::Medrops<Params, MGP_t, Omnigrasper, medrops::GPPolicy<Params>, policy_opt_t, RewardFunction> cp_system;
#endif

    cp_system.learn(20, 30);

    return 0;
}
