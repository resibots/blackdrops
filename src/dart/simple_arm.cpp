#include <limbo/limbo.hpp>

#include <boost/program_options.hpp>

#include <robot_dart/position_control.hpp>
#include <robot_dart/robot_dart_simu.hpp>

#ifdef GRAPHIC
#include <robot_dart/graphics.hpp>
#endif

#include <blackdrops/blackdrops.hpp>
#include <blackdrops/gp_model.hpp>
#include <blackdrops/model/gp/kernel_lf_opt.hpp>
#include <blackdrops/model/multi_gp.hpp>
#include <blackdrops/model/multi_gp/multi_gp_parallel_opt.hpp>

#include <blackdrops/policy/nn_policy.hpp>

#include <utils/utils.hpp>

struct Params {
#ifdef GRAPHIC
    struct graphics : robot_dart::defaults::graphics {
    };
#endif

    struct blackdrops {
        BO_PARAM(size_t, action_dim, 4);
        BO_PARAM(size_t, state_full_dim, 8);
        BO_PARAM(size_t, model_input_dim, 8);
        BO_PARAM(size_t, model_pred_dim, 4);
        BO_PARAM(size_t, rollout_steps, 39);
        BO_DYN_PARAM(double, boundary);
        BO_DYN_PARAM(bool, verbose);
    };

    struct gp_model {
        BO_PARAM(double, noise, 0.01);
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

    struct opt_rprop : public limbo::defaults::opt_rprop {
        BO_PARAM(int, iterations, 300);
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
        // BO_PARAM_ARRAY(double, max_u, 3.0, 3.0, 3.0, 3.0);
        BO_PARAM_ARRAY(double, max_u, 1.0, 1.0, 1.0, 1.0);
        BO_PARAM_ARRAY(double, limits, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
        BO_DYN_PARAM(int, hidden_neurons);
        BO_PARAM(double, af, 1.0);
    };
};

struct RewardParams {
    struct kernel : public limbo::defaults::kernel {
        BO_PARAM(double, noise, 1e-12);
        BO_PARAM(bool, optimize_noise, false);
    };

    struct kernel_squared_exp_ard : public limbo::defaults::kernel_squared_exp_ard {
    };

    struct opt_rprop : public limbo::defaults::opt_rprop {
        BO_PARAM(int, iterations, 300);
    };

    struct opt_parallelrepeater : public limbo::defaults::opt_parallelrepeater {
        BO_PARAM(int, repeats, 3);
    };
};

namespace global {
    std::shared_ptr<robot_dart::Robot> global_robot;

    using policy_t = blackdrops::policy::NNPolicy<PolicyParams>;

    Eigen::VectorXd goal(3);

    using kernel_t = limbo::kernel::SquaredExpARD<Params>;
    using mean_t = limbo::mean::Data<Params>;

    using GP_t = limbo::model::GP<RewardParams, kernel_t, mean_t, blackdrops::model::gp::KernelLFOpt<RewardParams>>;
    GP_t reward_gp(4, 1);
}

Eigen::VectorXd get_robot_state(const std::shared_ptr<robot_dart::Robot>& robot, bool full = false)
{
    Eigen::VectorXd pos = robot->skeleton()->getPositions();
    size_t size = pos.size();
    if (full)
        size += pos.size();
    Eigen::VectorXd state(size);
    if (!full)
        state = pos;
    else {
        for (int i = 0; i < pos.size(); i++) {
            state(2 * i) = std::cos(pos(i));
            state(2 * i + 1) = std::sin(pos(i));
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

        std::vector<double> params(4, 1.0);
        for (int i = 0; i < to_state.size(); i++)
            params[i] = to_state(i);
        robot_simu_t simu(params, simulated_robot);
        simu.run(2);

        auto bd = simulated_robot->skeleton()->getBodyNode("arm_link_5");
        Eigen::VectorXd eef = bd->getTransform().translation();
        double s_c_sq = 0.2 * 0.2;
        double dee = (eef - global::goal).squaredNorm();

        return std::exp(-0.5 / s_c_sq * dee);
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

struct SimpleArm {
    template <typename Policy, typename Reward>
    std::vector<std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd>> execute(const Policy& policy, const Reward& world, size_t steps, std::vector<double>& R, bool display = true)
    {
        static int n_iter = 0;
        std::vector<std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd>> res;
        Eigen::VectorXd pp = policy.params();
        std::vector<double> params(pp.size());
        Eigen::VectorXd::Map(params.data(), pp.size()) = pp;

        double t = 4.0;

        std::shared_ptr<robot_dart::Robot> simulated_robot = global::global_robot->clone();
        simulated_robot->fix_to_world();
        simulated_robot->set_position_enforced(true);

        std::vector<Eigen::VectorXd> coms, qs;

        class PolicyControl : public robot_dart::RobotControl {
        public:
            using robot_t = std::shared_ptr<robot_dart::Robot>;

            PolicyControl() {}
            PolicyControl(const std::vector<double>& ctrl, robot_t robot)
                : robot_dart::RobotControl(ctrl, robot)
            {
                size_t _start_dof = 0;
                if (!_robot->fixed_to_world()) {
                    _start_dof = 6;
                }
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
            }

            void update(double t)
            {
                _t = t;
                set_commands();
            }

            void set_commands()
            {
                double dt = 0.1;
                // double ds = 0.1;

                if (_t == 0.0 || (_t - _prev_time) >= dt) {
                    Eigen::VectorXd commands = _policy.next(get_robot_state(_robot, true));
                    Eigen::VectorXd q = get_robot_state(_robot);
                    qs->push_back(q);
                    coms->push_back(commands);

                    assert(_dof == (size_t)commands.size());
                    _robot->skeleton()->setCommands(commands);
                    _prev_commands = commands;
                    _prev_time = _t; //int(_t / dt) * dt;
                }
                else
                    _robot->skeleton()->setCommands(_prev_commands);
            }

            global::policy_t _policy;
            std::vector<Eigen::VectorXd>* coms;
            std::vector<Eigen::VectorXd>* qs;

        protected:
            double _prev_time;
            double _t;
            Eigen::VectorXd _prev_commands;
        };

#ifdef GRAPHIC
        using robot_simu_t = robot_dart::RobotDARTSimu<robot_dart::robot_control<PolicyControl>, robot_dart::graphics<robot_dart::Graphics<Params>>>;
#else
        using robot_simu_t = robot_dart::RobotDARTSimu<robot_dart::robot_control<PolicyControl>>;
#endif

        robot_simu_t simu(params, simulated_robot);
        // simulation step different from sampling rate -- we need a stable simulation
        simu.set_step(0.001);

        simu.controller()._policy = policy;
        simu.controller().coms = &coms;
        simu.controller().qs = &qs;

        // Add goal marker
        Eigen::Vector6d goal_pose = Eigen::Vector6d::Zero();
        goal_pose.tail(3) = global::goal;
        // pose, dims, type, mass, color, name
        simu.add_ellipsoid(goal_pose, {0.1, 0.1, 0.1}, "fixed", 1., dart::Color::Green(1.0), "goal_marker");
        // remove collisions from goal marker
        simu.world()->getSkeleton("goal_marker")->getRootBodyNode()->setCollidable(false);

        R = std::vector<double>();

        simu.run(t);

        ActualReward actual_reward;

        for (size_t j = 0; j < qs.size() - 1; j++) {
            size_t id = j; // * step;
            Eigen::VectorXd init(Params::blackdrops::model_pred_dim());
            init = qs[id];

            Eigen::VectorXd init_full(Params::blackdrops::model_input_dim());
            for (int i = 0; i < init.size(); i++) {
                init_full(2 * i) = std::cos(init(i));
                init_full(2 * i + 1) = std::sin(init(i));
            }

            Eigen::VectorXd u = coms[id];
            Eigen::VectorXd final(Params::blackdrops::model_pred_dim());
            final = qs[id + 1];
            double r = actual_reward(final);
            global::reward_gp.add_sample(final, limbo::tools::make_vector(r));
            R.push_back(r);
            res.push_back(std::make_tuple(init_full, u, final - init));
            // std::cout << final.transpose() << ": " << r << std::endl;
        }

        if (!policy.random() && display) {
            double rr = std::accumulate(R.begin(), R.end(), 0.0);
            std::cout << "Reward: " << rr << std::endl;
        }

        global::reward_gp.optimize_hyperparams();
        std::cout << "Learned the new reward function..." << std::endl;

        // Dump rewards
        int eval = 1000;
        Eigen::VectorXd limits(4);
        limits << M_PI, M_PI / 2.0, M_PI / 2.0, M_PI / 2.0;
        std::vector<Eigen::VectorXd> rvs = random_vectors(limits.size(), eval, limits);
        // std::vector<Eigen::VectorXd> rvs = global::reward_gp.samples();

        double mse = 0.0;
        std::ofstream ofs("reward_" + std::to_string(n_iter) + ".dat");
        for (size_t i = 0; i < rvs.size(); i++) {
            Eigen::VectorXd to_state = rvs[i];
            // Eigen::VectorXd eef = get_eef(to_state);
            // double de = (eef - global::goal).norm();
            // ofs<<"0 0 0 ";
            for (int j = 0; j < to_state.size(); j++)
                ofs << to_state[j] << " ";
            double r_b = world(to_state, to_state, to_state, true);
            double r_w = actual_reward(to_state);
            ofs << r_b << " " << r_w << std::endl;
            mse += (r_b - r_w) * (r_b - r_w);
        }
        ofs.close();
        std::cout << "MSE: " << mse / double(eval) << std::endl;
        n_iter++;

        return res;
    }

    template <typename Policy, typename Model, typename Reward>
    void execute_dummy(const Policy& policy, const Model& model, const Reward& world, size_t steps, std::vector<double>& R, bool display = true) const
    {
        R = std::vector<double>();
        // init state
        Eigen::VectorXd init = Eigen::VectorXd::Zero(Params::blackdrops::model_pred_dim());

        for (size_t j = 0; j < steps; j++) {
            Eigen::VectorXd query_vec(Params::blackdrops::model_input_dim() + Params::blackdrops::action_dim());
            Eigen::VectorXd init_full(Params::blackdrops::model_input_dim());
            for (int i = 0; i < init.size(); i++) {
                init_full(2 * i) = std::cos(init(i));
                init_full(2 * i + 1) = std::sin(init(i));
            }

            Eigen::VectorXd u = policy.next(init_full);
            query_vec.head(Params::blackdrops::model_input_dim()) = init_full;
            query_vec.tail(Params::blackdrops::action_dim()) = u;

            Eigen::VectorXd mu;
            Eigen::VectorXd sigma;
            std::tie(mu, sigma) = model.predictm(query_vec);

            Eigen::VectorXd final = init + mu;

            double r = world(init, mu, final, true);
            R.push_back(r);
            init = final;

            // std::cout << final.transpose() << " ---> " << r << std::endl;
        }
        // std::cout << "------------------" << std::endl;
    }

    template <typename Policy, typename Model, typename Reward>
    double predict_policy(const Policy& policy, const Model& model, const Reward& world, size_t steps) const
    {

        double reward = 0.0;
        // init state
        Eigen::VectorXd init = Eigen::VectorXd::Zero(Params::blackdrops::model_pred_dim());
        // init(5) = 0.58;
        for (size_t j = 0; j < steps; j++) {
            Eigen::VectorXd query_vec(Params::blackdrops::model_input_dim() + Params::blackdrops::action_dim());
            Eigen::VectorXd init_full(Params::blackdrops::model_input_dim());

            for (int i = 0; i < init.size(); i++) {
                init_full(2 * i) = std::cos(init(i));
                init_full(2 * i + 1) = std::sin(init(i));
            }

            Eigen::VectorXd u = policy.next(init_full);
            query_vec.head(Params::blackdrops::model_input_dim()) = init_full;
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
    double operator()(const Eigen::VectorXd& from_state, const Eigen::VectorXd& action, const Eigen::VectorXd& to_state, bool certain = false) const
    {
        Eigen::VectorXd mu;
        double s;
        std::tie(mu, s) = global::reward_gp.query(to_state);
        if (certain || !Params::opt_cmaes::handle_uncertainty())
            return mu(0);

        return std::max(0., gaussian_rand(mu(0), std::sqrt(s)));
    }
};

void init_simu(const std::string& robot_file)
{
    global::global_robot = std::make_shared<robot_dart::Robot>(robot_dart::Robot(robot_file, {}, "arm", true));

    // get goal position
    std::shared_ptr<robot_dart::Robot> simulated_robot = global::global_robot->clone();
    simulated_robot->fix_to_world();
    simulated_robot->set_position_enforced(true);

    // #ifdef GRAPHIC
    //     using robot_simu_t = robot_dart::RobotDARTSimu<robot_dart::robot_control<robot_dart::PositionControl>, robot_dart::graphics<robot_dart::Graphics<Params>>>;
    // #else
    using robot_simu_t = robot_dart::RobotDARTSimu<robot_dart::robot_control<robot_dart::PositionControl>>;
    // #endif

    // here set goal position
    std::vector<double> params = {M_PI / 4.0, M_PI / 8.0, M_PI / 8.0, M_PI / 8.0};

    robot_simu_t simu(params, simulated_robot);
    simu.run(2);
    auto bd = simulated_robot->skeleton()->getBodyNode("arm_link_5");
    global::goal = bd->getTransform().translation();

    std::cout << "Goal is: " << global::goal.transpose() << std::endl;
}

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
    namespace po = boost::program_options;
    po::options_description desc("Command line arguments");
    // clang-format off
    desc.add_options()("help,h", "Prints this help message")
                      ("hidden_neurons,n", po::value<int>(), "Number of hidden neurons in NN policy.")
                      ("boundary,b", po::value<double>(), "Boundary of the values during the optimization.")
                      ("max_evals,m", po::value<int>(), "Max function evaluations to optimize the policy.")
                      ("tolerance,t", po::value<double>(), "Maximum tolerance to continue optimizing the function.")
                      ("restarts,r", po::value<int>(), "Max number of restarts to use during optimization.")
                      ("elitism,e", po::value<int>(), "Elitism mode to use [0 to 3].")
                      ("uncertainty,u", po::bool_switch(&uncertainty)->default_value(false), "Enable uncertainty handling.")
                      ("threads,d", po::value<int>(), "Max number of threads used by TBB")
                      ("verbose,v", po::bool_switch(&verbose)->default_value(false), "Enable verbose mode.");
    // clang-format on

    try {
        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help")) {
            std::cout << desc << std::endl;
            return 0;
        }

        po::notify(vm);

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

    init_simu(std::string(RESPATH) + "/URDF/arm.urdf");

    using policy_opt_t = limbo::opt::Cmaes<Params>;

    using kernel_t = limbo::kernel::SquaredExpARD<Params>;
    using mean_t = limbo::mean::Constant<Params>;

    using GP_t = blackdrops::model::MultiGP<Params, limbo::model::GP, kernel_t, mean_t, blackdrops::model::multi_gp::MultiGPParallelLFOpt<Params, blackdrops::model::gp::KernelLFOpt<Params>>>;

    using MGP_t = blackdrops::GPModel<Params, GP_t>;

    blackdrops::BlackDROPS<Params, MGP_t, SimpleArm, blackdrops::policy::NNPolicy<PolicyParams>, policy_opt_t, RewardFunction> arm_system;

    arm_system.learn(2, 15, true);

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