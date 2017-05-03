#include <limbo/experimental/model/spgp.hpp>
#include <limbo/limbo.hpp>
#include <limbo/mean/constant.hpp>

#include <robot_dart/robot_dart_simu.hpp>
#include <robot_dart/position_control.hpp>
#ifdef GRAPHIC
#include <robot_dart/graphics.hpp>
#endif

#include <boost/program_options.hpp>

#include <medrops/cmaes.hpp>
#include <medrops/gp_model.hpp>
// #include <medrops/gp_multi_model.hpp>
#include <medrops/kernel_lf_opt.hpp>
#include <medrops/medrops.hpp>

#include <medrops/gp_policy.hpp>
#include <medrops/nn_policy.hpp>

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
    BO_DYN_PARAM(size_t, parallel_evaluations);

    struct options {
        BO_PARAM(bool, bounded, true);
    };

    struct medrops {
        BO_PARAM(size_t, action_dim, 3);
        BO_PARAM(size_t, state_full_dim, 12);
        BO_PARAM(size_t, model_input_dim, 9);
        BO_PARAM(size_t, model_pred_dim, 6);
        BO_PARAM(size_t, rollout_steps, 39);
        BO_DYN_PARAM(double, boundary);
        BO_DYN_PARAM(bool, verbose);
    };

    struct gp_model {
        BO_PARAM(double, noise, 0.01);
    };

    // struct model_spgp : public limbo::defaults::model_spgp {
    //     BO_PARAM(double, samples_percent, 10);
    //     BO_PARAM(double, jitter, 1e-5);
    //     BO_PARAM(int, min_m, 100);
    //     BO_PARAM(double, sig, 0.001);
    // };
    // struct model_gpmm : public limbo::defaults::model_gpmm {
    //     BO_PARAM(int, threshold, 300);
    // };

    struct mean_constant {
        BO_PARAM(double, constant, 0.0);
    };

    struct opt_nloptgrad : public limbo::defaults::opt_nloptgrad {
        BO_PARAM(int, iterations, 1000);
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
        BO_PARAM(int, iterations, 20000);
    };
};

struct PolicyParams {
    struct medrops : public Params::medrops {
    };

    struct nn_policy {
        BO_PARAM(size_t, state_dim, Params::medrops::model_input_dim());
        BO_PARAM(size_t, action_dim, Params::medrops::action_dim());
        BO_PARAM_ARRAY(double, max_u, 2.5, 44.7, 14.0);
        BO_DYN_PARAM(int, hidden_neurons);
    };

    struct gp_policy {
        BO_PARAM(size_t, state_dim, Params::medrops::model_input_dim());
        BO_PARAM(size_t, action_dim, Params::medrops::action_dim());
        BO_PARAM_ARRAY(double, max_u, 2.5, 44.7, 14.0);
        BO_PARAM(double, pseudo_samples, 10);
        BO_PARAM(double, noise, 1e-5);
    };

    struct kernel : public limbo::defaults::kernel {
        BO_PARAM(double, noise, gp_policy::noise());
    };

    struct kernel_squared_exp_ard : public limbo::defaults::kernel_squared_exp_ard {
    };
};

struct RewardParams {
    struct kernel : public limbo::defaults::kernel {
        BO_PARAM(double, noise, 0.001);
        BO_PARAM(bool, optimize_noise, false);
    };

    struct kernel_exp : public limbo::defaults::kernel_exp {
    };

    struct mean_constant {
        BO_PARAM(double, constant, 0.0);
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

// template <typename Policy>
// class PolicyControl : public robot_dart::RobotControl {
// public:
//     using robot_t = std::shared_ptr<robot_dart::Robot>;
//
//     PolicyControl() {}
//     PolicyControl(const std::vector<double>& ctrl, robot_t robot)
//         : robot_dart::RobotControl(ctrl, robot)
//     {
//         _robot->set_actuator_types(dart::dynamics::Joint::FORCE);
//         _prev_time = 0.0;
//         _t = 0.0;
//     }
//
//     void update(double t)
//     {
//         set_commands();
//         _t = t;
//     }
//
//     void set_commands()
//     {
//         double dt = 0.1;
//
//         if (_t == 0.0 || (_t - _prev_time) >= dt) {
//             Eigen::VectorXd pos = _robot->skeleton()->getPositions();
//             Eigen::VectorXd vel = _robot->skeleton()->getVelocities();
//             Eigen::VectorXd state(vel.size() + pos.size());
//             state.head(vel.size()) = vel;
//             state.tail(pos.size()) = pos;
//             Eigen::VectorXd commands = _policy.next(state);
//             data::vels.push_back(vel);
//             data::poses.push_back(pos);
//             data::coms.push_back(commands);
//             // std::cout << "state: " << vel.transpose() << " " << pos.transpose() << std::endl;
//             // std::cout << "command: " << commands.transpose() << std::endl;
//             assert(_dof == (size_t)commands.size());
//             _robot->skeleton()->setCommands(commands);
//             _prev_commands = commands;
//             _prev_time = _t;
//         }
//         else
//             _robot->skeleton()->setCommands(_prev_commands);
//     }
//
//     Policy _policy;
//
// protected:
//     double _prev_time;
//     double _t;
//     Eigen::VectorXd _prev_commands;
// };

namespace global {
    std::shared_ptr<robot_dart::Robot> global_robot;
#ifndef GPPOLICY
    using policy_t = medrops::NNPolicy<PolicyParams>;
#else
    using policy_t = medrops::GPPolicy<PolicyParams>;
#endif
    // #ifdef GRAPHIC
    //     using robot_simu_t = robot_dart::RobotDARTSimu<robot_dart::robot_control<PolicyControl<policy_t>>, robot_dart::desc<boost::fusion::vector<RobotTraj>>, robot_dart::graphics<robot_dart::Graphics<Params>>>;
    // #else
    //     using robot_simu_t = robot_dart::RobotDARTSimu<robot_dart::robot_control<PolicyControl<policy_t>>, robot_dart::desc<boost::fusion::vector<RobotTraj>>>;
    // #endif

    Eigen::VectorXd goal(3);

    using kernel_t = limbo::kernel::Exp<RewardParams>; //medrops::SquaredExpARD<Params>; //limbo::kernel::Exp<Params>;
    using mean_t = limbo::mean::Constant<RewardParams>;

    using GP_t = limbo::model::GP<RewardParams, kernel_t, mean_t>; //, medrops::KernelLFOpt<Params, limbo::opt::NLOptGrad<Params, nlopt::LD_SLSQP>>>;
    GP_t reward_gp(3, 1);
}

Eigen::VectorXd get_robot_state(const std::shared_ptr<robot_dart::Robot>& robot, bool full = false)
{
    // // Eigen::Vector3d size;
    // // size << 0, 0, 0.252;
    // // Eigen::VectorXd p = robot->body_trans("arm_3_sub")->getCOM();
    // Eigen::VectorXd pos = robot->skeleton()->getPositions();
    // auto bd = robot->skeleton()->getBodyNode("arm_3_sub");
    // Eigen::VectorXd p = bd->getCOM();
    // Eigen::VectorXd v = bd->getCOMLinearVelocity();
    // Eigen::VectorXd r(Params::medrops::model_input_dim());
    // r.segment(3, 3) = p;
    // r.head(3) = v;
    // r.tail(3) = pos;
    // return r;
    // std::cout << robot->skeleton()->getVelocityUpperLimits().transpose() << std::endl;
    Eigen::VectorXd pos = robot->skeleton()->getPositions();
    Eigen::VectorXd vel = robot->skeleton()->getVelocities();
    size_t size = vel.size() + pos.size();
    if (full)
        size += pos.size();
    Eigen::VectorXd state(size);
    state.head(vel.size()) = vel;
    if (!full)
        state.tail(pos.size()) = pos;
    else {
        for (int i = 0; i < pos.size(); i++) {
            state(vel.size() + 2 * i) = std::cos(pos(i));
            state(vel.size() + 2 * i + 1) = std::sin(pos(i));
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
        double s_c_sq = 0.1 * 0.1;
        // double s_c = 0.25 * 0.25;
        double dee = (eef - global::goal).squaredNorm();
        // Eigen::VectorXd ll(3);
        // ll << -M_PI, -1.80159265359, -1.82159265359;
        // Eigen::VectorXd ul(3);
        // ul << M_PI, 1.79840734641, 1.65840734641;

        // // JOINT LIMITS
        // Eigen::VectorXd ll(3);
        // ll << -M_PI, -1.2, -1.2;
        // Eigen::VectorXd ul(3);
        // ul << M_PI, 1.2, 1.2;
        // // double de = (ll - to_state).squaredNorm() + (ul - to_state).squaredNorm();
        //
        // for (size_t j = 0; j < 3; j++) {
        //     if (to_state[j] < ll[j]) {
        //         return -1.0;
        //     }
        //     if (to_state[j] > ul[j]) {
        //         return -1.0;
        //     }
        // }

        // return 1.0 - std::exp(-0.5 / s_c * de) + std::exp(-0.5 / s_c_sq * dee);
        return std::exp(-0.5 / s_c_sq * dee); // - std::exp(-0.5 / s_c * de);
    }
};

struct Omnigrasper {
    template <typename Policy, typename Reward>
    std::vector<std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd>> execute(const Policy& policy, const Reward& world, size_t steps, std::vector<double>& R, bool display = true)
    {
        std::vector<std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd>> res;
        Eigen::VectorXd pp = policy.params();
        std::vector<double> params(pp.size());
        Eigen::VectorXd::Map(params.data(), pp.size()) = pp;
        double t = 2.0, dt = 0.001;

        std::shared_ptr<robot_dart::Robot> simulated_robot = global::global_robot->clone();
        simulated_robot->fix_to_world();
        simulated_robot->set_position_enforced(true);

        // data::vels.clear();
        // data::poses.clear();
        // data::coms.clear();

        std::vector<Eigen::VectorXd> vels, poses, coms, qs;

        class PolicyControl : public robot_dart::RobotControl {
        public:
            using robot_t = std::shared_ptr<robot_dart::Robot>;

            PolicyControl() {}
            PolicyControl(const std::vector<double>& ctrl, robot_t robot)
                : robot_dart::RobotControl(ctrl, robot)
            {
                _robot->set_actuator_types(dart::dynamics::Joint::FORCE);
                _prev_time = 0.0;
                _prev_time_ds = 0.0;
                _t = 0.0;
            }

            void update(double t)
            {
                _t = t;
                set_commands();
            }

            void set_commands()
            {
                double dt = 0.05;
                // double ds = 0.1;

                if (_t == 0.0 || (_t - _prev_time) >= dt) {
                    Eigen::VectorXd commands = _policy.next(get_robot_state(_robot, true));
                    // if (_t == 0.0 || (_t - _prev_time_ds) >= ds) {
                    Eigen::VectorXd state = get_robot_state(_robot);
                    Eigen::VectorXd vel = state.head(3);
                    // Eigen::VectorXd pos = state.segment(3, 3);
                    // Eigen::VectorXd qq = state.tail(3);
                    Eigen::VectorXd pos = state.tail(3);
                    vels->push_back(vel);
                    poses->push_back(pos);
                    coms->push_back(commands);
                    // qs->push_back(qq);
                    _prev_time_ds = _t;
                    // }
                    assert(_dof == (size_t)commands.size());
                    _robot->skeleton()->setCommands(commands);
                    _prev_commands = commands;
                    _prev_time = int(_t / dt) * dt;
                }
                else
                    _robot->skeleton()->setCommands(_prev_commands);
            }

            global::policy_t _policy;
            std::vector<Eigen::VectorXd>* vels;
            std::vector<Eigen::VectorXd>* poses;
            std::vector<Eigen::VectorXd>* coms;
            std::vector<Eigen::VectorXd>* qs;

        protected:
            double _prev_time, _prev_time_ds;
            double _t;
            Eigen::VectorXd _prev_commands;
        };

#ifdef GRAPHIC
        using robot_simu_t = robot_dart::RobotDARTSimu<robot_dart::robot_control<PolicyControl>, robot_dart::graphics<robot_dart::Graphics<Params>>>;
#else
        using robot_simu_t = robot_dart::RobotDARTSimu<robot_dart::robot_control<PolicyControl>>;
#endif

        robot_simu_t simu(params, simulated_robot);
        simu.set_step(dt);
        // size_t total_steps = size_t(t / dt);
        // simu.set_desc_dump(total_steps / steps);
        // if (policy.random())
        //     simu.controller().set_random_policy();
        simu.controller()._policy = policy;
        simu.controller().vels = &vels;
        simu.controller().poses = &poses;
        simu.controller().coms = &coms;
        simu.controller().qs = &qs;

        R = std::vector<double>();

        simu.run(t);

        ActualReward actual_reward;

        // std::vector<Eigen::VectorXd> states;
        // simu.get_descriptor<RobotTraj>(states);

        // std::cout << "Yeah:" << std::endl;
        // for (size_t j = 1; j < states.size(); j++)
        // size_t step = total_steps / steps;
        // if (display)
        //     std::cout << "#: " << vels.size() << std::endl;
        for (size_t j = 0; j < vels.size() - 1; j++) {
            size_t id = j; // * step;
            Eigen::VectorXd init(Params::medrops::model_pred_dim());
            init.head(3) = vels[id];
            // init.segment(3, 3) = poses[id];
            init.tail(3) = poses[id]; //qs[id];
            Eigen::VectorXd init_full(Params::medrops::model_input_dim());
            init_full.head(3) = init.head(3);
            for (int i = 0; i < 3; i++) {
                init_full(3 + 2 * i) = std::cos(init(3 + i));
                init_full(3 + 2 * i + 1) = std::sin(init(3 + i));
            }
            Eigen::VectorXd u = coms[id];
            Eigen::VectorXd final(Params::medrops::model_pred_dim());
            final.head(3) = vels[id + 1];
            // final.segment(3, 3) = poses[id + 1];
            final.tail(3) = poses[id + 1]; //qs[id + 1];
            // Eigen::VectorXd init = states[j - 1].head(Params::medrops::model_input_dim());
            // Eigen::VectorXd u = states[j - 1].segment(Params::medrops::model_input_dim(), Params::medrops::action_dim());
            // Eigen::VectorXd final = states[j].head(Params::medrops::model_input_dim());
            // if (display) {
            //     std::cout << "state: " << init.transpose() << std::endl;
            //     std::cout << "command: " << u.transpose() << std::endl;
            //     // std::cout << poses[id].transpose() << " to " << poses[id + 1].transpose() << std::endl;
            //     // std::cout << "vel: " << vels[id].transpose() << std::endl;
            //     // std::cout << "vel: " << vels[id + 1].transpose() << std::endl;
            //     // std::cout << "my_vel: " << ((poses[id + 1] - poses[id]).array() / 0.05).transpose() << std::endl;
            // }
            // std::cout << "next state: " << final.transpose() << std::endl;
            double r = actual_reward(final.tail(3)); //world(init, u, final);
            global::reward_gp.add_sample(final.tail(3), limbo::tools::make_vector(r)); //, 0.001);
            R.push_back(r);
            res.push_back(std::make_tuple(init_full, u, final - init));
            std::cout << final.tail(3).transpose() << ": " << r << std::endl;
            if (r < 0)
                break;

            // Eigen::VectorXd ll(3);
            // ll << -M_PI, -1.2, -1.2;
            // Eigen::VectorXd ul(3);
            // ul << M_PI, 1.2, 1.2;
            // bool br = false;
            // for (size_t j = 0; j < 3; j++) {
            //     if (poses[id + 1][j] < ll[j]) {
            //         br = true;
            //         break;
            //     }
            //     if (poses[id + 1][j] > ul[j]) {
            //         br = true;
            //         break;
            //     }
            // }
            // if (br)
            //     break;
        }
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
        Eigen::VectorXd init = Eigen::VectorXd::Zero(Params::medrops::model_pred_dim());
        // init(5) = 0.58;
        for (size_t j = 0; j < steps; j++) {
            Eigen::VectorXd query_vec(Params::medrops::model_input_dim() + Params::medrops::action_dim());
            Eigen::VectorXd init_full(Params::medrops::model_input_dim());
            init_full.head(3) = init.head(3);
            for (int i = 0; i < 3; i++) {
                init_full(3 + 2 * i) = std::cos(init(3 + i));
                init_full(3 + 2 * i + 1) = std::sin(init(3 + i));
            }
            // init.tail(Params::medrops::model_input_dim()) = init.tail(Params::medrops::model_input_dim()).unaryExpr([](double x) { return angle_dist(0,x); });
            Eigen::VectorXd u = policy.next(init_full);
            query_vec.head(Params::medrops::model_input_dim()) = init_full;
            query_vec.tail(Params::medrops::action_dim()) = u;

            Eigen::VectorXd mu;
            Eigen::VectorXd sigma;
            std::tie(mu, sigma) = model.predictm(query_vec);

            Eigen::VectorXd final = init + mu;
            // std::cout << init.transpose() << " to " << final.transpose() << " dx: " << mu.transpose() << std::endl;
            // std::cout << "state: " << init.transpose() << std::endl;
            // std::cout << "command: " << u.transpose() << std::endl;
            // final.tail(Params::medrops::model_input_dim()) = final.tail(Params::medrops::model_input_dim()).unaryExpr([](double x) { return angle_dist(0,x); });

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
            Eigen::VectorXd init = Eigen::VectorXd::Zero(Params::medrops::model_pred_dim());
            // init(5) = 0.58;
            for (size_t j = 0; j < steps; j++) {
                Eigen::VectorXd query_vec(Params::medrops::model_input_dim() + Params::medrops::action_dim());
                Eigen::VectorXd init_full(Params::medrops::model_input_dim());
                init_full.head(3) = init.head(3);
                for (int i = 0; i < 3; i++) {
                    init_full(3 + 2 * i) = std::cos(init(3 + i));
                    init_full(3 + 2 * i + 1) = std::sin(init(3 + i));
                }
                Eigen::VectorXd u = policy.next(init_full);
                query_vec.head(Params::medrops::model_input_dim()) = init_full;
                query_vec.tail(Params::medrops::action_dim()) = u;

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
                // final.tail(Params::medrops::model_input_dim()) = final.tail(Params::medrops::model_input_dim()).unaryExpr([](double x) { return angle_dist(0,x); });

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
        // double de = (to_state.segment(3, 3) - global::goal).squaredNorm();
        // double de = 0.0;
        // for (size_t i = 0; i < 3; i++) {
        //     double dx = angle_dist(to_state(3 + i), global::goal(i));
        //     de += dx * dx;
        // }

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
    // #ifdef GRAPHIC
    //     using robot_simu_t = robot_dart::RobotDARTSimu<robot_dart::robot_control<robot_dart::PositionControl>, robot_dart::graphics<robot_dart::Graphics<Params>>>;
    // #else
    //     using robot_simu_t = robot_dart::RobotDARTSimu<robot_dart::robot_control<robot_dart::PositionControl>>;
    // #endif
    std::vector<double> params(3, 1.0);
    // params[0] = -2.5;
    // params[2] = 1.0;

    robot_simu_t simu(params, simulated_robot);
    simu.run(2);
    auto bd = simulated_robot->skeleton()->getBodyNode("arm_3_sub");
    global::goal = bd->getCOM();
    // global::goal = get_robot_state(simulated_robot).segment(3, 3);
    std::cout << "Goal is: " << global::goal.transpose() << std::endl;
    // global::goal = Eigen::VectorXd(3);
    // global::goal << 1, 1, 1;
}

BO_DECLARE_DYN_PARAM(size_t, Params, parallel_evaluations);
BO_DECLARE_DYN_PARAM(int, PolicyParams::nn_policy, hidden_neurons);
BO_DECLARE_DYN_PARAM(double, Params::medrops, boundary);
BO_DECLARE_DYN_PARAM(bool, Params::medrops, verbose);

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
    namespace po = boost::program_options;
    po::options_description desc("Command line arguments");
    desc.add_options()("help,h", "Prints this help message")("parallel_evaluations,p", po::value<int>(), "Number of parallel monte carlo evaluations for policy reward estimation.")("hidden_neurons,n", po::value<int>(), "Number of hidden neurons in NN policy.")("boundary,b", po::value<double>(), "Boundary of the values during the optimization.")("max_evals,m", po::value<int>(), "Max function evaluations to optimize the policy.")("tolerance,t", po::value<double>(), "Maximum tolerance to continue optimizing the function.")("restarts,r", po::value<int>(), "Max number of restarts to use during optimization.")("elitism,e", po::value<int>(), "Elitism mode to use [0 to 3].")("uncertainty,u", po::bool_switch(&uncertainty)->default_value(false), "Enable uncertainty handling.")("threads,d", po::value<int>(), "Max number of threads used by TBB")("verbose,v", po::bool_switch(&verbose)->default_value(false), "Enable verbose mode.");

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
            Params::medrops::set_boundary(c);
            Params::opt_cmaes::set_lbound(-c);
            Params::opt_cmaes::set_ubound(c);
        }
        else {
            Params::medrops::set_boundary(0);
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

    Params::medrops::set_verbose(verbose);
    Params::opt_cmaes::set_handle_uncertainty(uncertainty);

    std::cout << std::endl;
    std::cout << "Cmaes parameters:" << std::endl;
    std::cout << "  max_fun_evals = " << Params::opt_cmaes::max_fun_evals() << std::endl;
    std::cout << "  fun_tolerance = " << Params::opt_cmaes::fun_tolerance() << std::endl;
    std::cout << "  restarts = " << Params::opt_cmaes::restarts() << std::endl;
    std::cout << "  elitism = " << Params::opt_cmaes::elitism() << std::endl;
    std::cout << "  handle_uncertainty = " << Params::opt_cmaes::handle_uncertainty() << std::endl;
    std::cout << "  boundary = " << Params::medrops::boundary() << std::endl;
    std::cout << "  tbb threads = " << threads << std::endl;
    std::cout << std::endl;

    const char* env_p = std::getenv("RESIBOTS_DIR");
    // initilisation of the simulation and the simulated robot
    if (env_p) //if the environment variable exists
        init_simu(std::string(std::getenv("RESIBOTS_DIR")) + "/share/arm_models/URDF/omnigrasper_3dof.urdf");
    else //if it does not exist, we might be running this on the cluster
        init_simu("/nfs/hal01/kchatzil/Workspaces/ResiBots/share/arm_models/URDF/omnigrasper_3dof.urdf");

    using policy_opt_t = limbo::opt::CustomCmaes<Params>;

    using kernel_t = limbo::kernel::SquaredExpARD<Params>;
    using mean_t = limbo::mean::Constant<Params>;

    using GP_t = limbo::model::GP<Params, kernel_t, mean_t, medrops::KernelLFOpt<Params, limbo::opt::NLOptGrad<Params, nlopt::LD_SLSQP>>>;
    // using SPGP_t = limbo::model::SPGP<Params, kernel_t, mean_t>;

    // #ifdef SPGPS
    //     using GPMM_t = limbo::model::GPMultiModel<Params, mean_t, GP_t, SPGP_t>;
    //     using MGP_t = medrops::GPModel<Params, GPMM_t>;
    // #else
    //     using MGP_t = medrops::GPModel<Params, GP_t>;
    // #endif

    using MGP_t = medrops::GPModel<Params, GP_t>;

#ifndef GPPOLICY
    medrops::Medrops<Params, MGP_t, Omnigrasper, medrops::NNPolicy<PolicyParams>, policy_opt_t, RewardFunction> cp_system;
#else
    medrops::Medrops<Params, MGP_t, Omnigrasper, medrops::GPPolicy<PolicyParams>, policy_opt_t, RewardFunction> cp_system;
#endif

    cp_system.learn(20, 20, true);

    return 0;
}
