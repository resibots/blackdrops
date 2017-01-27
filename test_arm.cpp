#include <limbo/experimental/model/spgp.hpp>
#include <limbo/limbo.hpp>
#include <limbo/mean/constant.hpp>

#include <robot_dart/robot_dart_simu.hpp>
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
    BO_PARAM(size_t, action_dim, 5);
    BO_PARAM(size_t, state_full_dim, 10);
    BO_PARAM(size_t, model_input_dim, 10);
    BO_PARAM(size_t, model_pred_dim, 10);

    BO_DYN_PARAM(size_t, parallel_evaluations);
    BO_DYN_PARAM(bool, verbose);

    struct options {
        BO_PARAM(bool, bounded, true);
    };

    struct medrops {
        BO_PARAM(size_t, rollout_steps, 40);
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
        BO_PARAM(int, state_dim, 10);
        BO_PARAM(double, max_u, 40.0);
    };

    struct nn_policy {
        BO_PARAM(int, state_dim, 10);
        BO_PARAM(double, max_u, 40.0);
        BO_DYN_PARAM(int, hidden_neurons);
    };

    struct gp_policy { //: public medrops::defaults::gp_policy_defaults{
        BO_PARAM(double, max_u, 40.0); //max action
        BO_PARAM(double, pseudo_samples, 10);
        BO_PARAM(double, noise, 0.01);
        BO_PARAM(int, state_dim, 10);
    };

    struct mean_constant {
        BO_PARAM(double, constant, 0.0);
    };

    struct opt_nloptgrad : public limbo::defaults::opt_nloptgrad {
        BO_PARAM(int, iterations, 1000);
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
        BO_PARAM(int, verbose, false);
        BO_PARAM(bool, fun_compute_initial, true);
        // BO_PARAM(double, fun_target, 30);
        BO_DYN_PARAM(double, ubound);
        BO_DYN_PARAM(double, lbound);
        BO_DYN_PARAM(int, lambda);

        BO_PARAM(double, a, -32.0);
        BO_PARAM(double, b, 0.0);
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

struct RobotTraj : public robot_dart::descriptors::DescriptorBase {
public:
    RobotTraj() {}

    template <typename Simu, typename robot>
    void operator()(Simu& simu, std::shared_ptr<robot> rob)
    {
        Eigen::VectorXd pos = rob->skeleton()->getPositions();
        Eigen::VectorXd vel = rob->skeleton()->getVelocities();
        Eigen::VectorXd coms = rob->skeleton()->getCommands();
        Eigen::VectorXd state(vel.size() + pos.size() + coms.size());
        state.head(vel.size()) = vel;
        state.segment(vel.size(), pos.size()) = pos;
        state.tail(coms.size()) = coms;
        // std::cout << vel.transpose() << std::endl;
        // std::cout << pos.transpose() << std::endl;
        // std::cout << coms.transpose() << std::endl;
        // std::cout << state.transpose() << std::endl;
        _traj.push_back(state);
    }

    void get(std::vector<Eigen::VectorXd>& results)
    {
        results = _traj;
    }

protected:
    std::vector<Eigen::VectorXd> _traj;
};

namespace data {
    std::vector<Eigen::VectorXd> vels, poses, coms;
}

template <typename Policy>
class PolicyControl : public robot_dart::RobotControl {
public:
    using robot_t = std::shared_ptr<robot_dart::Robot>;

    PolicyControl() {}
    PolicyControl(const std::vector<double>& ctrl, robot_t robot)
        : robot_dart::RobotControl(ctrl, robot)
    {
        _robot->set_actuator_types(dart::dynamics::Joint::FORCE);
        _prev_time = 0.0;
        _t = 0.0;
    }

    void update(double t)
    {
        set_commands();
        _t = t;
    }

    void set_commands()
    {
        double dt = 0.1;

        if (_t == 0.0 || (_t - _prev_time) >= dt) {
            Eigen::VectorXd pos = _robot->skeleton()->getPositions();
            Eigen::VectorXd vel = _robot->skeleton()->getVelocities();
            Eigen::VectorXd state(vel.size() + pos.size());
            state.head(vel.size()) = vel;
            state.tail(pos.size()) = pos;
            Eigen::VectorXd commands = _policy.next(state);
            data::vels.push_back(vel);
            data::poses.push_back(pos);
            data::coms.push_back(commands);
            // std::cout << "state: " << vel.transpose() << " " << pos.transpose() << std::endl;
            // std::cout << "command: " << commands.transpose() << std::endl;
            assert(_dof == (size_t)commands.size());
            _robot->skeleton()->setCommands(commands);
            _prev_commands = commands;
            _prev_time = _t;
        }
        else
            _robot->skeleton()->setCommands(_prev_commands);
    }

    Policy _policy;

protected:
    double _prev_time;
    double _t;
    Eigen::VectorXd _prev_commands;
};

namespace global {
    std::shared_ptr<robot_dart::Robot> global_robot;
#ifndef GPPOLICY
    using policy_t = medrops::SFNNPolicy<Params>;
#else
    using policy_t = medrops::GPPolicy<Params>;
#endif
#ifdef GRAPHIC
    using robot_simu_t = robot_dart::RobotDARTSimu<robot_dart::robot_control<PolicyControl<policy_t>>, robot_dart::desc<boost::fusion::vector<RobotTraj>>, robot_dart::graphics<robot_dart::Graphics<Params>>>;
#else
    using robot_simu_t = robot_dart::RobotDARTSimu<robot_dart::robot_control<PolicyControl<policy_t>>, robot_dart::desc<boost::fusion::vector<RobotTraj>>>;
#endif

    Eigen::VectorXd goal(5);
}

struct Omnigrasper {
    template <typename Policy, typename Reward>
    std::vector<std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd>> execute(const Policy& policy, const Reward& world, size_t steps, std::vector<double>& R, bool display = true)
    {
        std::vector<std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd>> res;
        Eigen::VectorXd pp = policy.params();
        std::vector<double> params(pp.size());
        Eigen::VectorXd::Map(params.data(), pp.size()) = pp;
        double t = 4.0, dt = 0.001;

        std::shared_ptr<robot_dart::Robot> simulated_robot = global::global_robot->clone();
        simulated_robot->fix_to_world();
        simulated_robot->set_position_enforced(true);

        data::vels.clear();
        data::poses.clear();
        data::coms.clear();

        global::robot_simu_t simu(params, simulated_robot);
        simu.set_step(dt);
        size_t total_steps = size_t(t / dt);
        simu.set_desc_dump(total_steps / steps);
        // if (policy.random())
        //     simu.controller().set_random_policy();
        simu.controller()._policy = policy;

        R = std::vector<double>();

        simu.run(t);

        std::vector<Eigen::VectorXd> states;
        simu.get_descriptor<RobotTraj>(states);

        // std::cout << "Yeah:" << std::endl;
        // for (size_t j = 1; j < states.size(); j++)
        // size_t step = total_steps / steps;
        for (size_t j = 0; j < steps; j++) {
            size_t id = j; // * step;
            Eigen::VectorXd init(Params::model_input_dim());
            init.head(5) = data::vels[id];
            init.tail(5) = data::poses[id];
            Eigen::VectorXd u = data::coms[id];
            Eigen::VectorXd final(Params::model_input_dim());
            final.head(5) = data::vels[id + 1];
            final.tail(5) = data::poses[id + 1];
            // Eigen::VectorXd init = states[j - 1].head(Params::model_input_dim());
            // Eigen::VectorXd u = states[j - 1].segment(Params::model_input_dim(), Params::action_dim());
            // Eigen::VectorXd final = states[j].head(Params::model_input_dim());
            // std::cout << "state: " << init.transpose() << std::endl;
            // std::cout << "command: " << u.transpose() << std::endl;
            // std::cout << "next state: " << final.transpose() << std::endl;
            double r = world(init, u, final);
            R.push_back(r);
            res.push_back(std::make_tuple(init, u, final - init));
        }

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
        Eigen::VectorXd init = Eigen::VectorXd::Zero(Params::model_input_dim());
        for (size_t j = 0; j < steps; j++) {
            Eigen::VectorXd query_vec(Params::model_input_dim() + Params::action_dim());
            // init.tail(Params::model_input_dim()) = init.tail(Params::model_input_dim()).unaryExpr([](double x) { return angle_dist(0,x); });
            Eigen::VectorXd u = policy.next(init);
            query_vec.head(Params::model_input_dim()) = init;
            query_vec.tail(Params::action_dim()) = u;

            Eigen::VectorXd mu;
            Eigen::VectorXd sigma;
            std::tie(mu, sigma) = model.predictm(query_vec);

            Eigen::VectorXd final = init + mu;
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
            Eigen::VectorXd init = Eigen::VectorXd::Zero(Params::model_input_dim());
            for (size_t j = 0; j < steps; j++) {
                Eigen::VectorXd query_vec(Params::model_input_dim() + Params::action_dim());
                Eigen::VectorXd u = policy.next(init);
                query_vec.head(Params::model_input_dim()) = init;
                query_vec.tail(Params::action_dim()) = u;

                Eigen::VectorXd mu;
                Eigen::VectorXd sigma;
                std::tie(mu, sigma) = model.predictm(query_vec);

                if (Params::opt_cmaes::handle_uncertainty()) {
                    sigma = sigma.array();
                }
                else {
                    sigma = sigma.array().sqrt();
                }

                for (int i = 0; i < mu.size(); i++) {
                    double s = gaussian_rand(mu(i), sigma(i));
                    mu(i) = std::max(mu(i) - sigma(i),
                        std::min(s, mu(i) + sigma(i)));
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
        double s_c_sq = 0.25 * 0.25;
        // double de = (to_state.tail(5) - global::goal).squaredNorm();
        double de = 0.0;
        for (size_t i = 0; i < 3; i++) {
            double dx = angle_dist(to_state(5 + i), global::goal(i));
            de += dx * dx;
        }
        // std::cout << to_state.tail(5).transpose() << " vs " << global::goal.transpose() << std::endl;
        // std::cout << de << std::endl;

        return std::exp(-0.5 / s_c_sq * de);
    }
};

void init_simu(const std::string& robot_file)
{
    global::global_robot = std::make_shared<robot_dart::Robot>(robot_dart::Robot(robot_file, {}, "arm", true));
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
    namespace po = boost::program_options;
    po::options_description desc("Command line arguments");
    desc.add_options()("help,h", "Prints this help message")("parallel_evaluations,p", po::value<int>(), "Number of parallel monte carlo evaluations for policy reward estimation.")("hidden_neurons,n", po::value<int>(), "Number of hidden neurons in NN policy.")("boundary,b", po::value<double>(), "Boundary of the values during the optimization.")("lambda,l", po::value<int>(), "Initial population in CMA-ES (-1 to default)")("max_evals,m", po::value<int>(), "Max function evaluations to optimize the policy.")("tolerance,t", po::value<double>(), "Maximum tolerance to continue optimizing the function.")("restarts,r", po::value<int>(), "Max number of restarts to use during optimization.")("elitism,e", po::value<int>(), "Elitism mode to use [0 to 3].")("uncertainty,u", po::bool_switch(&uncertainty)->default_value(false), "Enable uncertainty handling.")("threads,d", po::value<int>(), "Max number of threads used by TBB")("verbose,v", po::bool_switch(&verbose)->default_value(false), "Enable verbose mode.");

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

    global::goal << 1, 1, 1, 1, 1;

    init_simu("/home/kchatzil/Workspaces/ResiBots/robots/robot_simu/robot_dart/res/models/omnigrasper.urdf");

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

    cp_system.learn(1, 15);

    return 0;
}
