#include <limbo/experimental/model/spgp.hpp>
#include <limbo/limbo.hpp>
#include <limbo/mean/constant.hpp>

#include <robot_dart/robot_dart_simu.hpp>

#ifdef GRAPHIC
#include <robot_dart/graphics.hpp>
#endif

#include <boost/program_options.hpp>

#include <blackdrops/cmaes.hpp>
#include <blackdrops/gp_model.hpp>
#include <blackdrops/gp_multi_model.hpp>
#include <spt/poegp.hpp>
#include <spt/poegp_lf_opt.hpp>
#include <blackdrops/kernel_lf_opt.hpp>
#include <blackdrops/blackdrops.hpp>
#include <blackdrops/parallel_gp.hpp>

#include <blackdrops/nn_policy.hpp>

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

    struct blackdrops {
        BO_PARAM(size_t, action_dim, 6);
        BO_PARAM(size_t, state_full_dim, 18);
        BO_PARAM(size_t, model_input_dim, 18);
        BO_PARAM(size_t, model_pred_dim, 18);
        BO_PARAM(size_t, rollout_steps, 100);
        BO_DYN_PARAM(double, boundary);
        BO_DYN_PARAM(bool, verbose);
    };

    struct gp_model {
        BO_PARAM(double, noise, 0.01);
    };

    struct spt_poegp : public spt::defaults::spt_poegp {
        BO_PARAM(int, leaf_size, 100);
        BO_PARAM(double, tau, 0.05);
    };

    struct model_gpmm : public limbo::defaults::model_gpmm {
        BO_PARAM(int, threshold, 198);
    };

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

    struct opt_rprop : public limbo::defaults::opt_rprop {
    };

    struct opt_parallelrepeater : public limbo::defaults::opt_parallelrepeater {
        BO_PARAM(int, repeats, 2);
    };
};

struct PolicyParams {
    struct blackdrops : public Params::blackdrops {
    };

    struct nn_policy {
        BO_PARAM(size_t, state_dim, Params::blackdrops::model_input_dim());
        BO_PARAM(size_t, action_dim, Params::blackdrops::action_dim());
        BO_PARAM_ARRAY(double, max_u, 120.0, 90.0, 60.0, 90.0, 60.0, 30.0);
        BO_DYN_PARAM(int, hidden_neurons);
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
    std::shared_ptr<robot_dart::Robot> global_robot;

    using policy_t = blackdrops::NNPolicy<PolicyParams>;
}

Eigen::VectorXd get_robot_state(const std::shared_ptr<robot_dart::Robot>& robot)
{
    Eigen::VectorXd pos = robot->skeleton()->getPositions();
    Eigen::VectorXd vel = robot->skeleton()->getVelocities();
    size_t size = vel.size() + pos.size() - 6;

    Eigen::VectorXd vels = Eigen::VectorXd::Zero(9);
    Eigen::VectorXd poses = Eigen::VectorXd::Zero(9);

    vels.head(3) << vel(3), vel(5), vel(1);
    vels.tail(6) = vel.tail(6);

    poses.head(3) << pos(3), pos(5), pos(1);
    poses.tail(6) = pos.tail(6);

    Eigen::VectorXd state(size);
    state.head(vels.size()) = vels;
    state.tail(poses.size()) = poses;

    return state;
}

struct HalfCheetah {
    template <typename Policy, typename Reward>
    std::vector<std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd>> execute(const Policy& policy, const Reward& world, size_t steps, std::vector<double>& R, bool display = true)
    {
        std::vector<std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd>> res;
        Eigen::VectorXd pp = policy.params();
        std::vector<double> params(pp.size());
        Eigen::VectorXd::Map(params.data(), pp.size()) = pp;
        double t = 5.0, dt = 0.001;

        std::shared_ptr<robot_dart::Robot> simulated_robot = global::global_robot->clone();
        simulated_robot->skeleton()->setPosition(5, 0.6);
        simulated_robot->set_position_enforced(true);

        std::vector<Eigen::VectorXd> vels, poses, coms;

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
                _policy.set_params(Eigen::VectorXd::Map(ctrl.data(), ctrl.size()));
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
                    // Eigen::VectorXd commands = _policy.next(get_robot_state(_robot, true));
                    // if (_t == 0.0 || (_t - _prev_time_ds) >= ds) {
                    Eigen::VectorXd state = get_robot_state(_robot);
                    Eigen::VectorXd vel = state.head(9);
                    Eigen::VectorXd pos = state.tail(9);
                    Eigen::VectorXd commands = _policy.next(state);

                    vels->push_back(vel);
                    poses->push_back(pos);
                    coms->push_back(commands);

                    _prev_time_ds = _t;
                    // }
                    assert(_dof == (size_t)commands.size() + 6);
                    Eigen::VectorXd cm(_dof);
                    cm.head(6) = Eigen::VectorXd::Zero(6);
                    cm.tail(commands.size()) = commands;
                    _robot->skeleton()->setCommands(cm);
                    _prev_commands = cm;
                    _prev_time = int(_t / dt) * dt;
                }
                else
                    _robot->skeleton()->setCommands(_prev_commands);
            }

            global::policy_t _policy;
            std::vector<Eigen::VectorXd>* vels;
            std::vector<Eigen::VectorXd>* poses;
            std::vector<Eigen::VectorXd>* coms;

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
        simu.add_floor(200.0);

#ifdef GRAPHIC
        simu.graphics()->free_camera();
#endif

        simu.controller()._policy = policy;
        simu.controller().vels = &vels;
        simu.controller().poses = &poses;
        simu.controller().coms = &coms;

        R = std::vector<double>();

        simu.run(t);

        for (size_t j = 0; j < vels.size() - 1; j++) {
            // R.push_back(vels[j](0));
            Eigen::VectorXd state(vels[j].size() + poses[j].size());
            state.head(vels[j].size()) = vels[j];
            state.tail(poses[j].size()) = poses[j];

            Eigen::VectorXd to_state(vels[j + 1].size() + poses[j + 1].size());
            to_state.head(vels[j + 1].size()) = vels[j + 1];
            to_state.tail(poses[j + 1].size()) = poses[j + 1];

            Eigen::VectorXd action = coms[j];

            R.push_back(world(state, action, to_state));

            res.push_back(std::make_tuple(state, action, to_state - state));
            // std::cout << state.transpose() << " with " << action.transpose() << " -> " << to_state.transpose() << std::endl;
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
        Eigen::VectorXd init = Eigen::VectorXd::Zero(Params::blackdrops::model_input_dim());
        init(10) = 0.6;
        // init(5) = 0.58;
        for (size_t j = 0; j < steps; j++) {
            Eigen::VectorXd query_vec(Params::blackdrops::model_input_dim() + Params::blackdrops::action_dim());

            Eigen::VectorXd u = policy.next(init);
            query_vec.head(Params::blackdrops::model_input_dim()) = init;
            query_vec.tail(Params::blackdrops::action_dim()) = u;

            Eigen::VectorXd mu;
            Eigen::VectorXd sigma;
            std::tie(mu, sigma) = model.predictm(query_vec);

            Eigen::VectorXd final = init + mu;

            double r = world(init, u, final);
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
            Eigen::VectorXd init = Eigen::VectorXd::Zero(Params::blackdrops::model_input_dim());
            init(10) = 0.6;
            // init(5) = 0.58;
            for (size_t j = 0; j < steps; j++) {
                Eigen::VectorXd query_vec(Params::blackdrops::model_input_dim() + Params::blackdrops::action_dim());

                Eigen::VectorXd u = policy.next(init);
                query_vec.head(Params::blackdrops::model_input_dim()) = init;
                query_vec.tail(Params::blackdrops::action_dim()) = u;

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
        return to_state(0);
    }
};

void init_simu(const std::string& robot_file)
{
    global::global_robot = std::make_shared<robot_dart::Robot>(robot_dart::Robot(robot_file, {}, "half_cheetah", true));
}

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

    const char* env_p = std::getenv("RESIBOTS_DIR");
    // initilisation of the simulation and the simulated robot
    if (env_p) //if the environment variable exists
        init_simu(std::string(std::getenv("RESIBOTS_DIR")) + "/share/robot_models/URDF/half_cheetah.urdf");
    else //if it does not exist, we might be running this on the cluster
        init_simu("/nfs/hal01/kchatzil/Workspaces/ResiBots/share/robot_models/URDF/half_cheetah.urdf");

    using policy_opt_t = limbo::opt::CustomCmaes<Params>;

    using kernel_t = limbo::kernel::SquaredExpARD<Params>;
    using mean_t = limbo::mean::Constant<Params>;

    // using GP_t = limbo::model::GP<Params, kernel_t, mean_t, blackdrops::KernelLFOpt<Params>>;
    // using SPGP_t = spt::POEGP<Params, kernel_t, mean_t, limbo::model::gp::POEKernelLFOpt<Params>>;
    using GP_t = blackdrops::ParallelGP<Params, limbo::model::GP, kernel_t, mean_t, blackdrops::KernelLFOpt<Params>>; //, limbo::opt::NLOptGrad<Params, nlopt::LD_SLSQP>>>;
    using SPGP_t = blackdrops::ParallelGP<Params, spt::POEGP, kernel_t, mean_t, limbo::model::gp::POEKernelLFOpt<Params>>;

#ifdef SPGPS
    using GPMM_t = limbo::model::GPMultiModel<Params, GP_t, SPGP_t>;
    using MGP_t = blackdrops::GPModel<Params, GPMM_t>;
#else
    using MGP_t = blackdrops::GPModel<Params, GP_t>;
#endif

    blackdrops::BlackDROPS<Params, MGP_t, HalfCheetah, blackdrops::NNPolicy<PolicyParams>, policy_opt_t, RewardFunction> cp_system;

    cp_system.learn(1, 20, true);

    return 0;
}
