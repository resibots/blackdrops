#include <limbo/limbo.hpp>
#include <limbo/mean/constant.hpp>

#include <dynamixel/safe_velocity_control.hpp>

#include <boost/program_options.hpp>

#include <blackdrops/gp_model.hpp>
// #include <blackdrops/gp_multi_model.hpp>
#include <blackdrops/blackdrops.hpp>
#include <blackdrops/model/gp/kernel_lf_opt.hpp>
#include <blackdrops/model/parallel_gp.hpp>

#include <blackdrops/policy/nn_policy.hpp>

#include <sstream>

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
    BO_PARAM(double, min_height, 0.1);

    struct gp_model {
        BO_PARAM(double, noise, 0.01);
    };

    struct blackdrops {
        BO_PARAM(size_t, action_dim, 4);
        BO_PARAM(size_t, state_full_dim, 12);
        BO_PARAM(size_t, model_input_dim, 8);
        BO_PARAM(size_t, model_pred_dim, 4);
        BO_PARAM(size_t, rollout_steps, 39);
        BO_PARAM(double, boundary, 1.0);
    };

    struct mean_constant {
        BO_PARAM(double, constant, 0.0);
    };

    struct kernel : public limbo::defaults::kernel {
        BO_PARAM(double, noise, gp_model::noise());
        BO_PARAM(bool, optimize_noise, false);
    };

    struct kernel_squared_exp_ard : public limbo::defaults::kernel_squared_exp_ard {
        BO_PARAM(double, sigma_sq, 0.2);
    };

    struct opt_nloptgrad : public limbo::defaults::opt_nloptgrad {
        BO_PARAM(int, iterations, 1000);
    };
};

struct PolicyParams {
    struct blackdrops : public Params::blackdrops {
    };

    struct nn_policy {
        BO_PARAM(size_t, state_dim, Params::blackdrops::model_input_dim());
        BO_PARAM(size_t, action_dim, Params::blackdrops::action_dim());
        BO_PARAM_ARRAY(double, max_u, 1.0, 1.0, 1.0, 1.0);
        BO_PARAM_ARRAY(double, limits, 1., 1., 1., 1., 1., 1., 1., 1.);
        BO_DYN_PARAM(int, hidden_neurons);
        BO_PARAM(double, af, 1.0);
    };
};

namespace global {
    using policy_t = blackdrops::policy::NNPolicy<PolicyParams>;

    Eigen::VectorXd goal(3);
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
        // std::cin >> reset;
        sleep(2);
        reset = false;
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

        return std::exp(-0.5 / s_c_sq * de);
    }
};

template <typename Policy>
double execute_policy(const Policy& policy)
{
    double t = 4.0, dt = 0.1;

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

    std::vector<double> R;

    ActualReward actual_reward;

    // if (display)
    std::cout << "#: " << q.size() << std::endl;
    std::cout << "init state: " << q[0].transpose() << std::endl;
    for (size_t id = 0; id < q.size() - 1; id++) {
        Eigen::VectorXd init(Params::blackdrops::model_pred_dim());
        init = q[id];
        Eigen::VectorXd init_full(Params::blackdrops::model_input_dim());
        for (int i = 0; i < 4; i++) {
            init_full(2 * i) = std::cos(init(i));
            init_full(2 * i + 1) = std::sin(init(i));
        }
        Eigen::VectorXd u = coms[id];
        Eigen::VectorXd final(Params::blackdrops::model_pred_dim());
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
        R.push_back(r);
        if (r < -0.9)
            break;
    }

    double rr = std::accumulate(R.begin(), R.end(), 0.0);
    std::cout << "Reward: " << rr << std::endl;

    return rr;
}

std::vector<std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd>> read_gp(const std::string& filename)
{
    std::vector<std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd>> data;

    std::string line;
    std::ifstream ifs(filename);
    // Params::blackdrops::model_input_dim()
    while (getline(ifs, line)) // same as: while (getline( myfile, line ).good())
    {
        std::stringstream ss(line);
        Eigen::VectorXd input(Params::blackdrops::model_input_dim());
        Eigen::VectorXd output(Params::blackdrops::model_pred_dim());
        Eigen::VectorXd action(Params::blackdrops::action_dim());

        for (size_t i = 0; i < Params::blackdrops::model_input_dim(); i++) {
            double d;
            ss >> d;
            input(i) = d;
        }

        for (size_t i = 0; i < Params::blackdrops::action_dim(); i++) {
            double d;
            ss >> d;
            action(i) = d;
        }

        for (size_t i = 0; i < Params::blackdrops::model_pred_dim(); i++) {
            double d;
            ss >> d;
            output(i) = d;
        }

        data.push_back(std::make_tuple(input, action, output));
    }

    return data;
}

BO_DECLARE_DYN_PARAM(int, PolicyParams::nn_policy, hidden_neurons);

int main(int argc, char** argv)
{
    std::string usb_port = "/dev/ttyUSB0";
    std::string directory = "./";

    namespace po = boost::program_options;
    po::options_description desc("Command line arguments");
    // clang-format off
    desc.add_options()("help,h", "Prints this help message")
    ("hidden_neurons,n", po::value<int>(), "Number of hidden neurons in NN policy.")
    ("usb,s", po::value<std::string>(), "Set the USB port for the robot")
    ("dir,d", po::value<std::string>(), "Set the USB port for the robot");
    // clang-format on

    try {
        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help")) {
            std::cout << desc << std::endl;
            return 0;
        }

        po::notify(vm);

        if (vm.count("usb")) {
            usb_port = vm["usb"].as<std::string>();
        }

        if (vm.count("dir")) {
            directory = vm["dir"].as<std::string>();
        }

        if (vm.count("hidden_neurons")) {
            int c = vm["hidden_neurons"].as<int>();
            if (c < 1)
                c = 1;
            PolicyParams::nn_policy::set_hidden_neurons(c);
        }
        else {
            PolicyParams::nn_policy::set_hidden_neurons(10);
        }
    }
    catch (po::error& e) {
        std::cerr << "[Exception caught while parsing command line arguments]: " << e.what() << std::endl;
        return 1;
    }

    if (!init_robot(usb_port)) {
        std::cerr << "Could not connect to the robot! Exiting...";
        exit(1);
    }

    std::cout << "Connected!" << std::endl;

    std::string random_file = directory + "random_policy_params_";
    std::string policy_file = directory + "policy_params_";
    std::string gp_data = directory + "gp_learn_";

    size_t random_trials = 1;
    size_t learning_trials = 15;

    using kernel_t = limbo::kernel::SquaredExpARD<Params>;
    using mean_t = limbo::mean::Constant<Params>;
    using GP_t = blackdrops::model::ParallelGP<Params, limbo::model::GP, kernel_t, mean_t, blackdrops::model::gp::KernelLFOpt<Params, limbo::opt::NLOptGrad<Params, nlopt::LD_SLSQP>>>;

    using MGP_t = blackdrops::GPModel<Params, GP_t>;

    double best_r = -std::numeric_limits<double>::max();
    size_t best = 0;
    // blackdrops::BlackDROPS<Params, MGP_t, Omnigrasper, blackdrops::NNPolicy<PolicyParams>, policy_opt_t, RewardFunction> cp_system;
    for (size_t i = 0; i < random_trials; i++) {
        std::cout << "Random trial #" << (i + 1) << std::endl;
        // Load policy
        Eigen::VectorXd policy_params;
        Eigen::read_binary(random_file + std::to_string(i) + ".bin", policy_params);
        blackdrops::policy::NNPolicy<PolicyParams> policy;
        policy.set_params(policy_params);
        double r = execute_policy(policy);
        if (r > best_r) {
            best_r = r;
            best = i;
        }

        // Sleep a bit
        sleep(2);
    }

    for (size_t i = 0; i < learning_trials; i++) {
        std::cout << "Learning trial #" << (i + 1) << std::endl;
        // Load policy
        Eigen::VectorXd policy_params;
        Eigen::read_binary(policy_file + std::to_string(i + 1) + ".bin", policy_params);
        blackdrops::policy::NNPolicy<PolicyParams> policy;
        policy.set_params(policy_params);

        // Load model points
        MGP_t gp;
        std::vector<std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd>> data = read_gp(gp_data + std::to_string(i) + ".dat");
        gp.learn(data, true);

        // Normalize policy
        // policy.normalize(gp);

        double r = execute_policy(policy);
        if (r > best_r) {
            best_r = r;
            best = i + 1;
        }

        // Sleep a bit
        sleep(2);
    }

    return 0;
}
