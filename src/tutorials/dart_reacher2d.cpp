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
#include <blackdrops/system/dart_system.hpp>

#include <blackdrops/policy/nn_policy.hpp>

#include <utils/dart_utils.hpp>
#include <utils/utils.hpp>

struct Params {
#ifdef GRAPHIC
    struct graphics : robot_dart::defaults::graphics {
    };
#endif

    struct blackdrops {
        BO_PARAM(size_t, action_dim, 2);
        BO_PARAM(size_t, model_input_dim, 6);
        BO_PARAM(size_t, model_pred_dim, 4);
        // BO_PARAM(double, dt, 0.1);
        // BO_PARAM(double, T, 3.0);
        BO_PARAM(double, dt, 0.02);
        BO_PARAM(double, T, 0.5);
        BO_DYN_PARAM(double, boundary);
        BO_DYN_PARAM(bool, verbose);
    };

    struct dart_system {
        BO_PARAM(double, sim_step, 0.001);
    };

    struct dart_policy_control {
        // BO_PARAM(dart::dynamics::Joint::ActuatorType, joint_type, dart::dynamics::Joint::SERVO);
        BO_PARAM(dart::dynamics::Joint::ActuatorType, joint_type, dart::dynamics::Joint::FORCE);
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
        // BO_PARAM_ARRAY(double, max_u, 10.0, 10.0);
        // BO_PARAM_ARRAY(double, limits, 10., 10., 1.0, 1.0, 1.0, 1.0);
        BO_PARAM_ARRAY(double, max_u, 200.0, 200.0);
        BO_PARAM_ARRAY(double, limits, 30., 70., 1.0, 1.0, 1.0, 1.0);
        BO_DYN_PARAM(int, hidden_neurons);
        BO_PARAM(double, af, 1.0);
    };
};

struct RewardParams {
    struct kernel : public limbo::defaults::kernel {
        BO_PARAM(double, noise, 1e-12);
        BO_PARAM(bool, optimize_noise, true);
    };

    struct kernel_squared_exp_ard : public limbo::defaults::kernel_squared_exp_ard {
    };

    struct mean_constant {
        BO_PARAM(double, constant, -1.);
    };

    struct opt_rprop : public limbo::defaults::opt_rprop {
        BO_PARAM(int, iterations, 300);
        BO_PARAM(double, eps_stop, 1e-4);
    };

    struct opt_parallelrepeater : public limbo::defaults::opt_parallelrepeater {
        BO_PARAM(int, repeats, 3);
    };
};

namespace global {
    std::shared_ptr<robot_dart::Robot> global_robot;
    dart::dynamics::SkeletonPtr global_floor;

    using policy_t = blackdrops::policy::NNPolicy<PolicyParams>;

    Eigen::VectorXd goal(3);

    using kernel_t = limbo::kernel::SquaredExpARD<RewardParams>;
    using mean_t = limbo::mean::Constant<RewardParams>;

    using GP_t = limbo::model::GP<RewardParams, kernel_t, mean_t, blackdrops::model::gp::KernelLFOpt<RewardParams>>;
    GP_t reward_gp(4, 1);
}

Eigen::VectorXd get_robot_state(const std::shared_ptr<robot_dart::Robot>& robot, bool full = false)
{
    Eigen::VectorXd vel = robot->skeleton()->getVelocities();
    Eigen::VectorXd pos = robot->skeleton()->getPositions();

    size_t size = pos.size() + vel.size();
    if (full)
        size += pos.size();

    Eigen::VectorXd state(size);

    state.head(vel.size()) = vel;

    if (!full) {
        state.tail(pos.size()) = pos;
    }
    else {
        for (int i = 0; i < pos.size(); i++) {
            state(vel.size() + 2 * i) = std::cos(pos(i));
            state(vel.size() + 2 * i + 1) = std::sin(pos(i));
        }
    }
    return state;
}

struct ActualReward {
    double operator()(const Eigen::VectorXd& from_state, const Eigen::VectorXd& action, const Eigen::VectorXd& to_state) const
    {
        using robot_simu_t = robot_dart::RobotDARTSimu<robot_dart::robot_control<robot_dart::PositionControl>>;

        std::shared_ptr<robot_dart::Robot> simulated_robot = global::global_robot->clone();

        std::vector<double> params(2, 1.0);
        for (int i = 0; i < 2; i++)
            params[i] = to_state(2 + i);
        robot_simu_t simu(params, simulated_robot);
        // Change gravity
        Eigen::VectorXd gravity(3);
        gravity << 0., -9.81, 0.;
        simu.world()->setGravity(gravity);
        simu.controller().control_root_joint(true);

        simu.run(2);

        auto bd = simulated_robot->skeleton()->getBodyNode("link2");
        Eigen::VectorXd eef = bd->getTransform().translation();
        // double s_c_sq = 0.2 * 0.2;
        double dee = (eef - global::goal).norm();
        double a = (action.array() / 200.0).square().sum();

        // return std::exp(-0.5 / s_c_sq * dee);
        return -dee - a;
        //  - action.squaredNorm() / (2. * PolicyParams::nn_policy::max_u(0) * PolicyParams::nn_policy::max_u(0));
    }
};

Eigen::VectorXd get_random_vector(size_t dim, Eigen::VectorXd bounds)
{
    Eigen::VectorXd rv = (limbo::tools::random_vector(dim).array() * 2 - 1);
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

struct PolicyControl : public blackdrops::system::BaseDARTPolicyControl<Params, global::policy_t> {
    using base_t = blackdrops::system::BaseDARTPolicyControl<Params, global::policy_t>;

    PolicyControl() : base_t() {}
    PolicyControl(const std::vector<double>& ctrl, base_t::robot_t robot) : base_t(ctrl, robot) {}

    Eigen::VectorXd get_state(const base_t::robot_t& robot, bool full) const
    {
        return get_robot_state(robot, full);
    }
};

struct DARTReacher : public blackdrops::system::DARTSystem<Params, PolicyControl> {
    using base_t = blackdrops::system::DARTSystem<Params, PolicyControl>;

    Eigen::VectorXd init_state() const
    {
        return Eigen::VectorXd::Zero(Params::blackdrops::model_pred_dim());
    }

    Eigen::VectorXd transform_state(const Eigen::VectorXd& original_state) const
    {
        Eigen::VectorXd ret = Eigen::VectorXd::Zero(Params::blackdrops::model_input_dim());
        ret.head(2) = original_state.head(2);
        for (int j = 0; j < 2; j++) {
            ret(2 + 2 * j) = std::cos(original_state(j + 2));
            ret(2 + 2 * j + 1) = std::sin(original_state(j + 2));
        }

        return ret;
    }

    std::shared_ptr<robot_dart::Robot> get_robot() const
    {
        std::shared_ptr<robot_dart::Robot> simulated_robot = global::global_robot->clone();

        return simulated_robot;
    }

    void add_extra_to_simu(base_t::robot_simu_t& simu) const
    {
        // Change gravity
        Eigen::VectorXd gravity(3);
        gravity << 0., -9.81, 0.;
        simu.world()->setGravity(gravity);
        // Add goal marker
        Eigen::Vector6d goal_pose = Eigen::Vector6d::Zero();
        goal_pose.tail(3) = global::goal;
        // pose, dims, type, mass, color, name
        simu.add_ellipsoid(goal_pose, {0.025, 0.025, 0.025}, "fixed", 1., dart::Color::Green(1.0), "goal_marker");
        // remove collisions from goal marker
        simu.world()->getSkeleton("goal_marker")->getRootBodyNode()->setCollidable(false);

        auto ground = global::global_floor->clone();
        Eigen::Vector6d floor_pose = Eigen::Vector6d::Zero();
        floor_pose(4) = -0.0125;
        simu.add_skeleton(ground, floor_pose, "fixed");

#ifdef GRAPHIC
        Eigen::Vector3d camera_pos = Eigen::Vector3d(0., 2., 0.);
        Eigen::Vector3d look_at = Eigen::Vector3d(0., 0., 0.);
        Eigen::Vector3d up = Eigen::Vector3d(0., 1., 0.);
        simu.graphics()->fixed_camera(camera_pos, look_at, up);
        // slow down visualization because 0.5 seconds is too fast
        simu.graphics()->set_render_period(0.03);
#endif
    }

    template <typename Policy, typename Reward>
    std::vector<std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd>> execute(const Policy& policy, const Reward& world, double T, std::vector<double>& R, bool display = true)
    {
        static int n_iter = 0;
        ActualReward actual_reward;
        std::vector<std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd>> ret = blackdrops::system::DARTSystem<Params, PolicyControl>::execute(policy, actual_reward, T, R, display);

        std::vector<Eigen::VectorXd> states = this->get_last_states();
        std::vector<Eigen::VectorXd> coms = this->get_last_commands();

        for (size_t i = 0; i < R.size(); i++) {
            double r = R[i];
            Eigen::VectorXd final = states[i + 1];
            Eigen::VectorXd action = coms[i];

            Eigen::VectorXd vec(4);
            vec.head(2) = final.tail(2);
            vec.tail(2) = action;
            // Eigen::VectorXd com = coms[i];
            // Eigen::VectorXd vec(final.size() + com.size());
            // vec.head(final.size()) = final;
            // vec.tail(com.size()) = com;
            // std::cout << com.transpose() << std::endl;
            global::reward_gp.add_sample(vec, limbo::tools::make_vector(r));
        }

        global::reward_gp.optimize_hyperparams();
        std::cout << "Learned the new reward function..." << std::endl;

        // Dump rewards
        int eval = 1000;
        Eigen::VectorXd limits(6);
        limits << 10., 10., M_PI, M_PI, 200., 200.;
        std::vector<Eigen::VectorXd> rvs = random_vectors(limits.size(), eval, limits);
        // std::vector<Eigen::VectorXd> rvs = global::reward_gp.samples();

        double mse = 0.0;
        std::ofstream ofs("reward_" + std::to_string(n_iter) + ".dat");
        for (size_t i = 0; i < rvs.size(); i++) {
            Eigen::VectorXd to_state = rvs[i].head(4);
            Eigen::VectorXd action = rvs[i].tail(2);
            // Eigen::VectorXd eef = get_eef(to_state);
            // double de = (eef - global::goal).norm();
            // ofs<<"0 0 0 ";
            for (int j = 0; j < to_state.size(); j++)
                ofs << to_state[j] << " ";
            double r_b = world(to_state, action, to_state, true);
            double r_w = actual_reward(to_state, action, to_state);
            ofs << r_b << " " << r_w << std::endl;
            mse += (r_b - r_w) * (r_b - r_w);
        }
        ofs.close();
        std::cout << "MSE: " << mse / double(eval) << std::endl;
        n_iter++;

        return ret;
    }
};

struct RewardFunction {
    double operator()(const Eigen::VectorXd& from_state, const Eigen::VectorXd& action, const Eigen::VectorXd& to_state, bool certain = false) const
    {
        Eigen::VectorXd mu;
        double s;
        Eigen::VectorXd vec(2 + action.size());
        vec.head(2) = to_state.tail(2);
        vec.tail(action.size()) = action;
        std::tie(mu, s) = global::reward_gp.query(vec);
        if (certain || !Params::opt_cmaes::handle_uncertainty())
            return mu(0);

        return std::min(0., gaussian_rand(mu(0), std::sqrt(s)));
    }
};

void init_simu(const std::string& robot_file)
{
    global::global_robot = std::make_shared<robot_dart::Robot>(utils::load_skel(robot_file, "arm"));
    Eigen::Isometry3d tf = global::global_robot->skeleton()->getRootBodyNode()->getParentJoint()->getTransformFromParentBodyNode();
    tf.translation() = Eigen::Vector3d(0., 0.01, 0.);
    global::global_robot->skeleton()->getRootBodyNode()->getParentJoint()->setTransformFromParentBodyNode(tf);

    global::global_floor = utils::load_skel(robot_file, "ground skeleton");

    global::goal = Eigen::Vector3d(0.1, 0.01, -0.1);

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

    init_simu(std::string(RESPATH) + "/skel/reacher2d.skel");

    using policy_opt_t = limbo::opt::Cmaes<Params>;

    using kernel_t = limbo::kernel::SquaredExpARD<Params>;
    using mean_t = limbo::mean::Constant<Params>;

    using GP_t = blackdrops::model::MultiGP<Params, limbo::model::GP, kernel_t, mean_t, blackdrops::model::multi_gp::MultiGPParallelLFOpt<Params, limbo::model::gp::KernelLFOpt<Params>>>;

    using MGP_t = blackdrops::GPModel<Params, GP_t>;

    blackdrops::BlackDROPS<Params, MGP_t, DARTReacher, global::policy_t, policy_opt_t, RewardFunction> reacher_system;

    reacher_system.learn(2, 15, true);

    ActualReward actual_reward;
    std::ofstream ofs("reward_points.dat");
    for (size_t i = 0; i < global::reward_gp.samples().size(); i++) {
        Eigen::VectorXd to_state = global::reward_gp.samples()[i];
        for (int j = 0; j < to_state.size(); j++)
            ofs << to_state[j] << " ";
        Eigen::VectorXd mu = global::reward_gp.mu(to_state);
        double r = actual_reward(to_state, to_state, to_state);
        ofs << mu(0) << " " << r;
        ofs << std::endl;
    }
    ofs.close();

    return 0;
}