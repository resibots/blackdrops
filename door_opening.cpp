#include <limbo/limbo.hpp>

#include <boost/program_options.hpp>

#include <robot_dart/robot_dart_simu.hpp>

#ifdef GRAPHIC
#include <robot_dart/graphics.hpp>
#endif

#include <blackdrops/cmaes.hpp>
#include <blackdrops/gp_model.hpp>
#include <blackdrops/gp_multi_model.hpp>
#include <blackdrops/multi_gp.hpp>
#include <blackdrops/multi_gp_whole_opt.hpp>
#include <blackdrops/parallel_gp.hpp>
#include <spt/poegp.hpp>
#include <spt/poegp_lf_opt.hpp>
#include <blackdrops/kernel_lf_opt.hpp>
#include <blackdrops/blackdrops.hpp>

#include <blackdrops/gp_policy.hpp>
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

    struct options {
        BO_PARAM(bool, bounded, true);
    };

    BO_PARAM(double, goal_pos, -M_PI / 2.0);

    struct blackdrops {
        BO_PARAM(size_t, action_dim, 5);
        BO_PARAM(size_t, state_full_dim, 11);
        BO_PARAM(size_t, model_input_dim, 11);
        BO_PARAM(size_t, model_pred_dim, 11);
        // TO-DO: See how many steps
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
        BO_PARAM_ARRAY(double, max_u, 3.0, 3.46, 1.67, 3.0, 3.11);
        BO_DYN_PARAM(int, hidden_neurons);
    };
};

namespace global {
    std::shared_ptr<robot_dart::Robot> global_robot, door_robot;
}

Eigen::VectorXd get_robot_state(const std::shared_ptr<robot_dart::Robot>& robot, const std::shared_ptr<robot_dart::Robot>& door)
{
    Eigen::VectorXd pos = robot->skeleton()->getPositions();
    Eigen::VectorXd vels = robot->skeleton()->getVelocities();
    size_t size = vels.size() + pos.size() + 1;
    Eigen::VectorXd state(size);
    state.head(vels.size()) = vels;
    state.segment(vels.size(), pos.size()) = pos;
    state.tail(1) = door->skeleton()->getPositions();

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
        // _policy.set_params(Eigen::VectorXd::Map(ctrl.data(), ctrl.size()));
    }

    void update(double t)
    {
        _t = t;
        set_commands();
    }

    void set_commands()
    {
        double dt = 0.05;

        // std::cout << _t << " vs " << _prev_time << " --> " << (_t - _prev_time) << std::endl;
        if (_t == 0.0 || (_t - _prev_time - dt) >= -1e-5) {
            // std::cout << "in" << std::endl;
            // std::cout << _t << " vs " << _prev_time << " --> " << (_t - _prev_time) << std::endl;
            Eigen::VectorXd state = get_robot_state(_robot, door);
            Eigen::VectorXd vel = state.head(5);
            Eigen::VectorXd pos = state.segment(5, 5);
            Eigen::VectorXd commands = policy.next(state);

            qs->push_back(pos);
            vels->push_back(vel);
            coms->push_back(commands);
            doors->push_back(state.tail(1)[0]);

            assert(_dof == (size_t)commands.size());
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
    std::vector<double>* doors;
    std::shared_ptr<robot_dart::Robot> door;
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
        double t = 5.0;

#ifndef GRAPHIC
        using robot_simu_t = robot_dart::RobotDARTSimu<robot_dart::robot_control<PolicyControl>>; //, robot_dart::collision<dart::collision::FCLCollisionDetector>>;
#else
        using robot_simu_t = robot_dart::RobotDARTSimu<robot_dart::robot_control<PolicyControl>, robot_dart::graphics<robot_dart::Graphics<Params>>>; //, robot_dart::collision<dart::collision::FCLCollisionDetector>, robot_dart::graphics<robot_dart::Graphics<Params>>>;
#endif

        Eigen::VectorXd ctrl_params = policy.params();

        std::vector<double> ctrl(ctrl_params.size(), 0.0);
        Eigen::VectorXd::Map(ctrl.data(), ctrl.size()) = ctrl_params;

        auto c_robot = global::global_robot->clone();
        c_robot->fix_to_world();
        c_robot->set_position_enforced(true);

        auto c_door = global::door_robot->clone();

        auto simu = robot_simu_t(ctrl, c_robot);
        simu.controller().policy = policy;
        simu.controller().door = c_door;

        Eigen::Vector6d pose = Eigen::VectorXd::Zero(6);
        pose.tail(3) = Eigen::Vector3d(-0.6, 0.0, 0.0);
        pose.head(3) = Eigen::Vector3d(0, 0, -dart::math::constants<double>::pi() / 2.0);
        simu.add_skeleton(c_door->skeleton(), pose, "fixed", "door");

        simu.set_step(0.01);
        simu.add_floor();

        std::vector<Eigen::VectorXd> qs, coms, vels;
        std::vector<double> doors;

        simu.controller().qs = &qs;
        simu.controller().vels = &vels;
        simu.controller().coms = &coms;
        simu.controller().doors = &doors;

#ifdef GRAPHIC
        simu.graphics()->fixed_camera(Eigen::Vector3d(1.0, 1.0, 1.5), Eigen::Vector3d(-0.6, 0.0, 0.0));
#endif

        simu.run(t);

        assert(doors.size() == 101);
        R = std::vector<double>();

        std::vector<std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd>> ret;

        for (size_t i = 0; i < doors.size() - 1; i++) {
            Eigen::VectorXd state(Params::blackdrops::model_input_dim());
            state.head(5) = vels[i];
            state.segment(5, 5) = qs[i];
            state.tail(1) = limbo::tools::make_vector(doors[i]);

            Eigen::VectorXd command = coms[i];

            Eigen::VectorXd to_state(Params::blackdrops::model_input_dim());
            to_state.head(5) = vels[i + 1];
            to_state.segment(5, 5) = qs[i + 1];
            to_state.tail(1) = limbo::tools::make_vector(doors[i + 1]);

            ret.push_back(std::make_tuple(state, command, to_state - state));
            R.push_back(world(state, command, to_state));
        }

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
        for (size_t j = 0; j < steps; j++) {
            Eigen::VectorXd query_vec(Params::blackdrops::model_input_dim() + Params::blackdrops::action_dim());

            Eigen::VectorXd u = policy.next(init);
            query_vec.head(Params::blackdrops::model_input_dim()) = init;
            query_vec.tail(Params::blackdrops::action_dim()) = u;

            Eigen::VectorXd mu;
            Eigen::VectorXd sigma;
            std::tie(mu, sigma) = model.predictm(query_vec);

            Eigen::VectorXd final = init + mu;

            double r = world(init, mu, final);
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
        for (size_t j = 0; j < steps; j++) {
            if (init.norm() > 50)
                break;
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
        double s_c_sq = 0.1 * 0.1;
        double da = angle_dist(Params::goal_pos(), to_state.tail(1)[0]);

        return std::exp(-0.5 / s_c_sq * da * da);
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

        _velocities = Eigen::VectorXd::Map(ctrl.data(), ctrl.size());
    }

    void update(double t)
    {
        set_commands();
    }

    void set_commands()
    {
        Eigen::VectorXd state = get_robot_state(_robot, door);
        Eigen::VectorXd vel = state.head(5);
        Eigen::VectorXd pos = state.segment(5, 5);

        qs->push_back(pos);
        vels->push_back(vel);
        doors->push_back(state.tail(1)[0]);

        assert(_dof == (size_t)_velocities.size());
        _robot->skeleton()->setCommands(_velocities);
    }

    std::vector<Eigen::VectorXd>* qs;
    std::vector<Eigen::VectorXd>* vels;
    std::vector<double>* doors;
    std::shared_ptr<robot_dart::Robot> door;

protected:
    Eigen::VectorXd _velocities;
};

struct MeanFunc {

    MeanFunc(int dim_out = 1) {}

    template <typename GP>
    Eigen::VectorXd operator()(const Eigen::VectorXd& v, const GP& gp) const
    {
        double dt = 0.05;

        using robot_simu_t = robot_dart::RobotDARTSimu<robot_dart::robot_control<VelocityControl>>; //, robot_dart::collision<dart::collision::FCLCollisionDetector>>;
        // #ifndef GRAPHIC
        //         using robot_simu_t = robot_dart::RobotDARTSimu<robot_dart::robot_control<VelocityControl>, robot_dart::collision<dart::collision::FCLCollisionDetector>>;
        // #else
        //         using robot_simu_t = robot_dart::RobotDARTSimu<robot_dart::robot_control<VelocityControl>, robot_dart::collision<dart::collision::FCLCollisionDetector>, robot_dart::graphics<robot_dart::Graphics<Params>>>;
        // #endif

        Eigen::VectorXd ctrl_params = v.tail(5);

        std::vector<double> ctrl(ctrl_params.size(), 0.0);
        Eigen::VectorXd::Map(ctrl.data(), ctrl.size()) = ctrl_params;

        auto c_robot = global::global_robot->clone();
        c_robot->fix_to_world();
        c_robot->set_position_enforced(true);

        auto c_door = global::door_robot->clone();

        auto simu = robot_simu_t(ctrl, c_robot);
        Eigen::VectorXd velocities = v.head(5);
        Eigen::VectorXd positions = v.segment(5, 5);
        c_robot->skeleton()->setVelocities(velocities);
        c_robot->skeleton()->setPositions(positions);
        simu.controller().door = c_door;

        Eigen::Vector6d pose = Eigen::VectorXd::Zero(6);
        pose.tail(3) = Eigen::Vector3d(-0.6, 0.0, 0.0);
        pose.head(3) = Eigen::Vector3d(0, 0, -dart::math::constants<double>::pi() / 2.0);
        simu.add_skeleton(c_door->skeleton(), pose, "fixed", "door");
        c_door->skeleton()->setPositions(v.segment(10, 1));

        simu.set_step(0.05);
        simu.add_floor();

        std::vector<Eigen::VectorXd> qs, vels;
        std::vector<double> doors;

        simu.controller().qs = &qs;
        simu.controller().vels = &vels;
        simu.controller().doors = &doors;

        // #ifdef GRAPHIC
        //         simu.graphics()->fixed_camera(Eigen::Vector3d(1.0, 1.0, 1.5), Eigen::Vector3d(-0.6, 0.0, 0.0));
        // #endif

        simu.run(dt + 0.01);

        // assert(doors.size() == 1);

        Eigen::VectorXd state = v.head(11);

        Eigen::VectorXd command = ctrl_params;

        Eigen::VectorXd to_state(Params::blackdrops::model_input_dim());
        to_state.head(5) = vels.back();
        to_state.segment(5, 5) = qs.back();
        to_state.tail(1) = limbo::tools::make_vector(doors.back());
        // std::cout << state.transpose() << " ---> " << to_state.transpose() << " with " << ctrl_params.transpose() << std::endl;

        return (to_state - state);
    }

    Eigen::VectorXd h_params() const { return _params; }

    void set_h_params(const Eigen::VectorXd& params)
    {
        _params = params;
    }

protected:
    Eigen::VectorXd _params;
};
#endif

void init_simu(const std::string& robot_file, const std::string& door_file)
{
    global::global_robot = std::make_shared<robot_dart::Robot>(robot_dart::Robot(robot_file, {}, "arm", true));
    global::door_robot = std::make_shared<robot_dart::Robot>(robot_dart::Robot(door_file, {}, "door", true));
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

    // Load robot files
    const char* env_p = std::getenv("RESIBOTS_DIR");
    // initilisation of the simulation and the simulated robot
    if (env_p) //if the environment variable exists
        init_simu(std::string(std::getenv("RESIBOTS_DIR")) + "/share/arm_models/URDF/omnigrasper_hook.urdf", std::string(std::getenv("RESIBOTS_DIR")) + "/share/robot_models/URDF/door.urdf");
    else //if it does not exist, we might be running this on the cluster
        init_simu("/nfs/hal01/kchatzil/Workspaces/ResiBots/share/arm_models/URDF/omnigrasper_hook.urdf", "/nfs/hal01/kchatzil/Workspaces/ResiBots/share/robot_models/URDF/door.urdf");

    using policy_opt_t = limbo::opt::CustomCmaes<Params>;

    using kernel_t = limbo::kernel::SquaredExpARD<Params>;
#ifndef MEAN
    using mean_t = limbo::mean::Constant<Params>;
#else
    using mean_t = MeanFunc;
#endif

#ifndef MODELIDENT
    using GP_t = blackdrops::ParallelGP<Params, limbo::model::GP, kernel_t, mean_t, blackdrops::KernelLFOpt<Params>>; //, limbo::opt::NLOptGrad<Params, nlopt::LD_SLSQP>>>;
    using SPGP_t = blackdrops::ParallelGP<Params, spt::POEGP, kernel_t, mean_t, limbo::model::gp::POEKernelLFOpt<Params>>;
#else
    using GP_t = blackdrops::MultiGP<Params, limbo::model::GP, kernel_t, mean_t, blackdrops::MultiGPWholeLFOpt<Params, limbo::opt::NLOptNoGrad<Params, nlopt::LN_SBPLX>>>;
    using SPGP_t = blackdrops::MultiGP<Params, spt::POEGP, kernel_t, mean_t, blackdrops::MultiGPWholeLFOpt<Params, limbo::opt::NLOptNoGrad<Params, nlopt::LN_SBPLX>, limbo::model::gp::POEKernelLFOpt<Params>>>;
#endif

#ifdef SPGPS
    using GPMM_t = limbo::model::GPMultiModel<Params, GP_t, SPGP_t>;
    using MGP_t = blackdrops::GPModel<Params, GPMM_t>;
#else
    using MGP_t = blackdrops::GPModel<Params, GP_t>;
#endif

    blackdrops::BlackDROPS<Params, MGP_t, Omnigrasper, blackdrops::NNPolicy<PolicyParams>, policy_opt_t, RewardFunction> omni_door;

    omni_door.learn(1, 15);

    return 0;
}
