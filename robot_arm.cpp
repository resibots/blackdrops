#include <limbo/experimental/model/spgp.hpp>
#include <limbo/limbo.hpp>
#include <limbo/mean/constant.hpp>

#include <medrops/safe_velocity_control.hpp>

#include <boost/program_options.hpp>

#include <medrops/cmaes.hpp>
// #include <medrops/exp_sq_ard.hpp>
#include <medrops/exp_ard_noise.hpp>
#include <medrops/gp.hpp>
#define MEDROPS_GP
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
    BO_PARAM(size_t, action_dim, 4);
    BO_PARAM(size_t, state_full_dim, 12);
    BO_PARAM(size_t, model_input_dim, 8);
    BO_PARAM(size_t, model_pred_dim, 4);

    BO_PARAM(double, min_height, 0.1);

    BO_DYN_PARAM(size_t, parallel_evaluations);
    BO_DYN_PARAM(bool, verbose);

    struct options {
        BO_PARAM(bool, bounded, true);
    };

    struct medrops {
        BO_PARAM(size_t, rollout_steps, 74);
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
        BO_PARAM(int, state_dim, 8);
        BO_PARAM_ARRAY(double, max_u, 1.0, 1.0, 1.0, 1.0);
    };

    struct nn_policy {
        BO_PARAM(int, state_dim, 8);
        BO_PARAM_ARRAY(double, max_u, 1.0, 1.0, 1.0, 1.0);
        BO_DYN_PARAM(int, hidden_neurons);
    };

    struct gp_policy {
        BO_PARAM_ARRAY(double, max_u, 1.0, 1.0, 1.0, 1.0);
        BO_PARAM(double, pseudo_samples, 10);
        BO_PARAM(double, noise, 0.01);
        BO_PARAM(int, state_dim, 8);
    };

    struct mean_constant {
        BO_PARAM(double, constant, 0.0);
    };

    struct opt_nloptgrad : public limbo::defaults::opt_nloptgrad {
        BO_PARAM(int, iterations, 1000);
    };

    struct kernel_squared_exp_ard : public limbo::defaults::kernel_squared_exp_ard {
        BO_PARAM(double, sigma_sq, 0.2);
    };

    struct kernel_exp : public limbo::defaults::kernel_exp {
        BO_PARAM(double, sigma_sq, 1);
        BO_PARAM(double, l, 1);
    };

    struct opt_rprop : public limbo::defaults::opt_rprop {
    };

    struct opt_parallelrepeater : public limbo::defaults::opt_parallelrepeater {
        BO_PARAM(int, repeats, 2);
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

struct GPParams {
    struct mean_constant {
        BO_PARAM(double, constant, 0.0);
    };

    struct kernel_squared_exp_ard : public limbo::defaults::kernel_squared_exp_ard {
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

namespace global {
#ifndef GPPOLICY
    using policy_t = medrops::SFNNPolicy<Params>;
#else
    using policy_t = medrops::GPPolicy<Params>;
#endif

    Eigen::VectorXd goal(3);

    using kernel_t = medrops::SquaredExpARDNoise<GPParams>; //limbo::kernel::Exp<Params>;
    using mean_t = limbo::mean::Data<GPParams>;

    using GP_t = medrops::GP<Params, kernel_t, mean_t, medrops::KernelLFOpt<Params>>; //, limbo::opt::NLOptGrad<Params, nlopt::LD_SLSQP>>>;
    GP_t reward_gp(4, 1);

    std::shared_ptr<dynamixel::SafeVelocityControl> robot_control;
}

Eigen::VectorXd get_eef(const Eigen::VectorXd& q)
{
    std::vector<double> tmp_q(q.size());
    Eigen::VectorXd::Map(tmp_q.data(), tmp_q.size()) = q;
    std::vector<double> tmp_eef = global::robot_control->get_eef(tmp_q);

    return Eigen::VectorXd::Map(tmp_eef.data(), tmp_eef.size());
}

bool init_robot(const std::string& usb_port)
{
    std::map<dynamixel::SafeVelocityControl::id_t, double> min_angles = {{1, 0.5}, {2, 1.3}, {3, 1.3}, {4, 1.3}};
    std::map<dynamixel::SafeVelocityControl::id_t, double> max_angles = {{1, 2 * M_PI - 0.5}, {2, M_PI + 1.84}, {3, M_PI + 1.84}, {4, M_PI + 1.84}};
    // conservative velocity limits
    std::map<dynamixel::SafeVelocityControl::id_t, double> max_velocities = {{1, 3}, {2, 3}, {3, 3}, {4, 3}};

    std::unordered_set<dynamixel::protocols::Protocol1::id_t> selected_servos = {1, 2, 3, 4};

    try {
        global::robot_control = std::make_shared<dynamixel::SafeVelocityControl>(usb_port, selected_servos, min_angles, max_angles, max_velocities, Params::min_height());
    }
    catch (dynamixel::errors::Error e) {
        std::cerr << "Dynamixel error:\n\t" << e.msg() << std::endl;
        return false;
    }

    Eigen::VectorXd q(4);
    q << M_PI / 4.0, M_PI / 8.0, M_PI / 8.0, M_PI / 8.0;
    global::goal = get_eef(q);
    std::cout << "Goal is: " << global::goal.transpose() << std::endl;

    std::vector<double> qq(4, 0.0);
    Eigen::VectorXd::Map(qq.data(), qq.size()) = q;
    global::robot_control->go_to_target(qq, 1e-2);
    std::cout << "This is the goal position..." << std::endl;
    char c;
    std::cin >> c;

    return true;
}

void reset_robot()
{
    bool reset = true;
    while (reset) {
        // move to initial position
        global::robot_control->init_position();
        std::cout << "Reset again? " << std::endl;
        std::cin >> reset;
    }
}

Eigen::VectorXd get_robot_state(const std::vector<double>& vec, bool full = false)
{
    size_t size = vec.size();
    if (full)
        size = size * 2;
    Eigen::VectorXd state(size);
    for (size_t i = 0; i < 4; i++)
        state(i) = vec[i];
    if (full) {
        for (int i = 0; i < 4; i++) {
            state(2 * i) = std::cos(vec[i]);
            state(2 * i + 1) = std::sin(vec[i]);
        }
    }
    return state;
}

struct ActualReward {
    double operator()(const Eigen::VectorXd& to_state) const
    {
        Eigen::VectorXd eef = get_eef(to_state);
        double s_c_sq = 0.2 * 0.2;
        double de = (eef - global::goal).squaredNorm();
        // double pen = -1.0;
        //
        // if (eef(2) < Params::min_height())
        //     return pen;
        //
        // std::map<dynamixel::SafeVelocityControl::id_t, double> min_angles = {{1, 0.0}, {2, M_PI / 2.0}, {3, M_PI / 2.0}, {4, M_PI / 2.0}};
        // std::map<dynamixel::SafeVelocityControl::id_t, double> max_angles = {{1, 2 * M_PI}, {2, 3 * M_PI / 2.0}, {3, 3 * M_PI / 2.0}, {4, 3 * M_PI / 2.0}};
        //
        // for (size_t i = 0; i < 4; i++) {
        //     if (to_state(i) < min_angles[i + 1] - M_PI)
        //         return pen;
        //     if (to_state(i) > max_angles[i + 1] - M_PI)
        //         return pen;
        // }

        // return -std::sqrt(de * de); //std::exp(-0.5 / s_c_sq * de);
        return std::exp(-0.5 / s_c_sq * de);
    }
};

Eigen::VectorXd get_random_vector(size_t dim, Eigen::VectorXd bounds)
{
    Eigen::VectorXd rv = (limbo::tools::random_vector(dim).array() * 2 - 1);
    // rv(0) *= 3; rv(1) *= 5; rv(2) *= 6; rv(3) *= M_PI; rv(4) *= 10;
    return rv.cwiseProduct(bounds);
}

std::vector<Eigen::VectorXd> random_vectors(size_t dim, size_t q, Eigen::VectorXd bounds)
{
    std::vector<Eigen::VectorXd> result(q);
    for (size_t i = 0; i < q; i++) {
        result[i] = get_random_vector(dim, bounds);
    }
    return result;
}

struct Omnigrasper {
    template <typename Policy, typename Reward>
    std::vector<std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd>> execute(const Policy& policy, const Reward& world, size_t steps, std::vector<double>& R, bool display = true)
    {
        static int n_iter = 0;
        std::vector<std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd>> res;
        Eigen::VectorXd pp = policy.params();
        double t = 4.0, dt = 0.05;

        // Recording data
        std::vector<Eigen::VectorXd> q, coms;

        // map for velocities --- defaults to zero
        std::map<dynamixel::protocols::Protocol1::id_t, double> velocities;
        velocities[1] = 0.0;
        velocities[2] = 0.0;
        velocities[3] = 0.0;
        velocities[4] = 0.0;

        bool limit_reached = false;

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
                auto actuators_state = global::robot_control->joint_angles();
                for (size_t i = 0; i < 4; i++)
                    actuators_state[i] = actuators_state[i] - M_PI;

                // convert to Eigen vectors
                Eigen::VectorXd full_state = get_robot_state(actuators_state, true);
                // std::cout << "f: " << full_state.transpose() << std::endl;
                Eigen::VectorXd state = get_robot_state(actuators_state);

                // Query policy for next commands
                Eigen::VectorXd commands = policy.next(full_state);
                for (size_t i = 0; i < commands.size(); i++) {
                    velocities[i + 1] = commands(i);
                }

                // std::cout << "st: " << state.transpose() << " com: " << commands.transpose() << std::endl;

                // Update statistics
                q.push_back(state);
                coms.push_back(commands);

                // // Send commands
                // global::robot_control->velocity_command(velocities);
            }
            // Send commands
            global::robot_control->velocity_command(velocities);
            // if (global::robot_control->enforce_joint_limits()) {
            //     std::cout << "Reached joint limits!" << std::endl;
            //     // limit_reached = true;
            //     //
            //     // auto actuators_state = global::robot_control->joint_angles();
            //     // for (size_t i = 0; i < 4; i++)
            //     //     actuators_state[i] = actuators_state[i] - M_PI;
            //     //
            //     // // convert to Eigen vectors
            //     // Eigen::VectorXd full_state = get_robot_state(actuators_state, true);
            //     // Eigen::VectorXd state = get_robot_state(actuators_state);
            //     //
            //     // // Query policy for next commands
            //     // Eigen::VectorXd commands = policy.next(full_state);
            //     //
            //     // // Update statistics
            //     // q.push_back(state);
            //     // coms.push_back(commands);
            //     //
            //     // break;
            // }
            total_elapsed = std::chrono::steady_clock::now() - start_time;
            // std::cout << "Elasped: " << total_elapsed.count() << std::endl;
        } while (total_elapsed.count() <= t);
        total_elapsed = std::chrono::steady_clock::now() - start_time;

        // reset_fail = true;
        // while (reset_fail) {
        //     try {
        //         // reset robot
        //         reset_robot();
        //         reset_fail = false;
        //     }
        //     catch (dynamixel::errors::Error e) {
        //         reset_fail = true;
        //         std::cerr << "Dynamixel error:\n\t" << e.msg() << std::endl;
        //         std::cout << "Did you reset the power? Just press any key..." << std::endl;
        //         char c;
        //         std::cin >> c;
        //         init_robot("/dev/ttyUSB0");
        //     }
        // }

        velocities[1] = 0.0;
        velocities[2] = 0.0;
        velocities[3] = 0.0;
        velocities[4] = 0.0;
        global::robot_control->velocity_command(velocities);

        R = std::vector<double>();

        ActualReward actual_reward;

        // if (display)
        std::cout << "#: " << q.size() << std::endl;
        std::cout << "init state: " << q[0].transpose() << std::endl;
        for (size_t id = 0; id < q.size() - 1; id++) {
            Eigen::VectorXd init(Params::model_pred_dim());
            init = q[id];
            Eigen::VectorXd init_full(Params::model_input_dim());
            for (int i = 0; i < 4; i++) {
                init_full(2 * i) = std::cos(init(i));
                init_full(2 * i + 1) = std::sin(init(i));
            }
            Eigen::VectorXd u = coms[id];
            Eigen::VectorXd final(Params::model_pred_dim());
            final = q[id + 1];

            // if (display) {
            // std::cout << "state: " << init.transpose() << std::endl;
            // std::cout << "command: " << u.transpose() << std::endl;
            // std::cout << poses[id].transpose() << " to " << poses[id + 1].transpose() << std::endl;
            // std::cout << "vel: " << vels[id].transpose() << std::endl;
            // std::cout << "vel: " << vels[id + 1].transpose() << std::endl;
            // std::cout << "my_vel: " << ((poses[id + 1] - poses[id]).array() / 0.05).transpose() << std::endl;
            // }
            // std::cout << "next state: " << final.transpose() << std::endl;
            double r = actual_reward(final); //world(init, u, final);
            // if (limit_reached && j == vels.size() - 2)
            //     r = -10;
            std::cout << u.transpose() << std::endl;
            std::cout << final.transpose() << ": " << r << std::endl;
            global::reward_gp.add_sample(final, limbo::tools::make_vector(r)); //, 0.001);
            R.push_back(r);
            res.push_back(std::make_tuple(init_full, u, final - init));
            if (r < -0.9)
                break;
        }

        if (!policy.random() && display) {
            double rr = std::accumulate(R.begin(), R.end(), 0.0);
            std::cout << "Reward: " << rr << std::endl;
        }

        global::reward_gp.optimize_hyperparams();

        // Dump rewards
        int eval = 10000;
        Eigen::VectorXd limits(4);
        limits << M_PI - 0.5, 1.84, 1.84, 1.84;
        std::vector<Eigen::VectorXd> rvs = random_vectors(limits.size(), eval, limits);
        // std::vector<Eigen::VectorXd> rvs = global::reward_gp.samples();

        std::ofstream ofs("reward_" + std::to_string(n_iter) + ".dat");
        for (size_t i = 0; i < rvs.size(); i++) {
            Eigen::VectorXd to_state = rvs[i];
            Eigen::VectorXd eef = get_eef(to_state);
            double de = (eef - global::goal).norm();
            // ofs<<"0 0 0 ";
            for (int j = 0; j < to_state.size(); j++)
                ofs << to_state[j] << " ";
            ofs << de << " " << world(to_state, to_state, to_state, true) << " " << actual_reward(to_state) << std::endl;
        }
        ofs.close();

        n_iter++;
        return res;
    }

    template <typename Policy, typename Model, typename Reward>
    void execute_dummy(const Policy& policy, const Model& model, const Reward& world, size_t steps, std::vector<double>& R, bool display = true) const
    {
        std::cout << "Dummy: " << std::endl;
        ActualReward actual_reward;
        R = std::vector<double>();
        // init state
        Eigen::VectorXd init = Eigen::VectorXd::Zero(Params::model_pred_dim());
        std::cout << "init state: " << init.transpose() << std::endl;
        for (size_t j = 0; j < steps; j++) {
            Eigen::VectorXd query_vec(Params::model_input_dim() + Params::action_dim());
            Eigen::VectorXd init_full(Params::model_input_dim());
            for (int i = 0; i < 4; i++) {
                init_full(2 * i) = std::cos(init(i));
                init_full(2 * i + 1) = std::sin(init(i));
            }
            // init.tail(Params::model_input_dim()) = init.tail(Params::model_input_dim()).unaryExpr([](double x) { return angle_dist(0,x); });
            Eigen::VectorXd u = policy.next(init_full);
            query_vec.head(Params::model_input_dim()) = init_full;
            query_vec.tail(Params::action_dim()) = u;

            Eigen::VectorXd mu;
            Eigen::VectorXd sigma;
            std::tie(mu, sigma) = model.predictm(query_vec);

            Eigen::VectorXd final = init + mu;

            double r = world(init, mu, final, true);
            std::cout << u.transpose() << std::endl;
            std::cout << final.transpose() << ": " << r << " -> " << actual_reward(final) << std::endl;
            R.push_back(r);
            init = final;
        }
        std::cout << "----------------------" << std::endl;
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
                for (int i = 0; i < 4; i++) {
                    init_full(2 * i) = std::cos(init(i));
                    init_full(2 * i + 1) = std::sin(init(i));
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
    double operator()(const Eigen::VectorXd& from_state, const Eigen::VectorXd& action, const Eigen::VectorXd& to_state, bool certain = false) const
    {
        Eigen::VectorXd mu;
        double s;
        std::tie(mu, s) = global::reward_gp.query(to_state);
        if (certain)
            return mu(0);

        return gaussian_rand(mu(0), std::sqrt(s));
        // ActualReward actual_reward;
        // return actual_reward(to_state);
    }
};

using kernel_t = medrops::SquaredExpARDNoise<Params>;
using mean_t = limbo::mean::Constant<Params>;

using GP_t = medrops::GP<Params, kernel_t, mean_t, medrops::KernelLFOpt<Params, limbo::opt::NLOptGrad<Params, nlopt::LD_SLSQP>>>;
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

    std::cout << "Connected!" << std::endl;

    // try {
    //     reset_robot();
    // }
    // catch (dynamixel::errors::Error e) {
    //     std::cerr << "Dynamixel error:\n\t" << e.msg() << std::endl;
    // }

    // while (true) {
    //     if (global::robot_control->enforce_joint_limits()) {
    //         std::cout << "Reached joint limits!" << std::endl;
    //         // break;
    //     }
    //
    //     auto actuators_state = global::robot_control->joint_angles();
    //     Eigen::VectorXd q(4);
    //     for (auto& a : actuators_state)
    //         a = a - M_PI;
    //     q = Eigen::VectorXd::Map(actuators_state.data(), actuators_state.size());
    //     Eigen::VectorXd eef = get_eef(q);
    //     std::cout << eef.transpose() << std::endl;
    //     // std::cout << "Angles: ";
    //     // for (auto a : actuators_state)
    //     //     std::cout << a - M_PI
    //     //               << " ";
    //     // std::cout << std::endl;
    // }

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

    cp_system.learn(3, 15, true);

    ActualReward actual_reward;
    std::ofstream ofs("reward_points.dat");
    for (size_t i = 0; i < global::reward_gp.samples().size(); i++) {
        Eigen::VectorXd to_state = global::reward_gp.samples()[i];
        for (int j = 0; j < to_state.size(); j++)
            ofs << to_state[j] << " ";
        Eigen::VectorXd mu = global::reward_gp.mu(to_state);
        double r = actual_reward(to_state);
        ofs << mu(0) << " " << r;
        ofs << std::endl;
    }
    ofs.close();

    return 0;
}
