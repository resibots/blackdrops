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

#include <blackdrops/policy/gp_policy.hpp>
#include <blackdrops/policy/nn_policy.hpp>

#include <blackdrops/reward/gp_reward.hpp>

#include <utils/cmd_args.hpp>
#include <utils/dart_utils.hpp>
#include <utils/utils.hpp>

struct Params {
#ifdef GRAPHIC
    struct graphics : robot_dart::defaults::graphics {
    };
#endif

    struct blackdrops : public ::blackdrops::defaults::blackdrops {
        BO_PARAM(size_t, action_dim, 2);
        // BO_PARAM(size_t, model_input_dim, 8);
        // BO_PARAM(size_t, model_pred_dim, 6);
        BO_PARAM(size_t, model_input_dim, 6);
        BO_PARAM(size_t, model_pred_dim, 4);
        // BO_PARAM(double, dt, 0.1);
        // BO_PARAM(double, T, 3.0);
        BO_PARAM(double, dt, 0.02);
        BO_PARAM(double, T, 0.5);
        BO_DYN_PARAM(double, boundary);
        BO_DYN_PARAM(bool, verbose);
        BO_DYN_PARAM(bool, stochastic);

        BO_PARAM(bool, stochastic_evaluation, true);
        BO_PARAM(int, num_evals, 1000);
        BO_DYN_PARAM(int, opt_evals);
        // BO_PARAM(int, opt_evals, 10);
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

        BO_PARAM(int, variant, aIPOP_CMAES);
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
        BO_PARAM(size_t, state_dim, Params::blackdrops::model_input_dim() + 2);
        BO_PARAM(size_t, action_dim, Params::blackdrops::action_dim());
        BO_PARAM_ARRAY(double, max_u, 200.0, 200.0);
        BO_PARAM_ARRAY(double, limits, 0.2, 0.2, 30., 70., 1.0, 1.0, 1.0, 1.0);
        BO_DYN_PARAM(int, hidden_neurons);
        BO_PARAM(double, af, 1.0);
    };

    struct gp_policy {
        BO_PARAM(size_t, state_dim, Params::blackdrops::model_input_dim() + 2);
        BO_PARAM(size_t, action_dim, Params::blackdrops::action_dim());
        BO_PARAM_ARRAY(double, max_u, 200.0, 200.0);
        BO_DYN_PARAM(int, pseudo_samples);
        BO_PARAM(double, noise, 0.01 * 0.01);
        BO_PARAM_ARRAY(double, limits, 0.2, 0.2, 30., 70., 1.0, 1.0, 1.0, 1.0);
    };

    struct kernel : public limbo::defaults::kernel {
        BO_PARAM(double, noise, gp_policy::noise());
    };

    struct kernel_squared_exp_ard : public limbo::defaults::kernel_squared_exp_ard {
    };
};

struct RewardParams : public blackdrops::reward_defaults {
    struct mean_constant {
        BO_PARAM(double, constant, -1.);
    };
};

namespace global {
    std::shared_ptr<robot_dart::Robot> global_robot;
    dart::dynamics::SkeletonPtr global_floor;

#ifndef GPPOLICY
    using policy_t = blackdrops::policy::NNPolicy<PolicyParams>;
#else
    using policy_t = blackdrops::policy::GPPolicy<PolicyParams>;
#endif

    Eigen::VectorXd goal(3);
} // namespace global

Eigen::VectorXd get_robot_state(const std::shared_ptr<robot_dart::Robot>& robot)
{
    Eigen::VectorXd vel = robot->skeleton()->getVelocities();
    Eigen::VectorXd pos = robot->skeleton()->getPositions();

    size_t size = pos.size() + vel.size();

    Eigen::VectorXd state(size);

    state.head(vel.size()) = vel;
    state.tail(pos.size()) = pos;

    return state;
}

Eigen::VectorXd get_random_vector(size_t dim, const Eigen::VectorXd& bounds)
{
    Eigen::VectorXd rv = (limbo::tools::random_vector(dim).array() * 2 - 1);
    return rv.cwiseProduct(bounds);
}

std::vector<Eigen::VectorXd> random_vectors(size_t dim, size_t q, const Eigen::VectorXd& bounds)
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

    Eigen::VectorXd get_state(const base_t::robot_t& robot) const
    {
        return get_robot_state(robot);
    }
};

struct DARTReacher : public blackdrops::system::DARTSystem<Params, PolicyControl, blackdrops::RolloutInfo> {
    using base_t = blackdrops::system::DARTSystem<Params, PolicyControl, blackdrops::RolloutInfo>;

    Eigen::VectorXd init_state() const
    {
        Eigen::VectorXd limits(Params::blackdrops::model_pred_dim());
        // limits << 0.2, 0.2, 0.005, 0.005, 0.1, 0.1;
        limits << 0.005, 0.005, 0.1, 0.1;

        Eigen::VectorXd state = get_random_vector(limits.size(), limits);

        return state;
    }

    blackdrops::RolloutInfo get_rollout_info() const
    {
        blackdrops::RolloutInfo info;
        info.init_state = this->init_state();
        info.t = 0;

        Eigen::VectorXd limits(2);
        limits << 0.2, 0.2;

        Eigen::VectorXd target = get_random_vector(limits.size(), limits);
        while (target.norm() >= 0.2) {
            target = get_random_vector(limits.size(), limits);
        }

        info.target = target;

        return info;
    }

    Eigen::VectorXd transform_state(const Eigen::VectorXd& original_state) const
    {
        Eigen::VectorXd ret = Eigen::VectorXd::Zero(Params::blackdrops::model_input_dim());
        // int st = 4;
        int st = 2;
        ret.head(st) = original_state.head(st);
        for (int j = 0; j < 2; j++) {
            ret(st + 2 * j) = std::cos(original_state(j + st));
            ret(st + 2 * j + 1) = std::sin(original_state(j + st));
        }

        return ret;
    }

    Eigen::VectorXd policy_transform(const Eigen::VectorXd& original_state, blackdrops::RolloutInfo* info) const
    {
        Eigen::VectorXd ret = Eigen::VectorXd::Zero(original_state.size() + 2);

        ret.head(2) = info->target;
        ret.tail(original_state.size()) = original_state;

        return ret;
    }

    void set_robot_state(const std::shared_ptr<robot_dart::Robot>& robot, const Eigen::VectorXd& state) const
    {
        robot->skeleton()->setPositions(state.tail(2));
        // robot->skeleton()->setVelocities(state.segment(2, 2));
        robot->skeleton()->setVelocities(state.head(2));
    }

    std::shared_ptr<robot_dart::Robot> get_robot() const
    {
        std::shared_ptr<robot_dart::Robot> simulated_robot = global::global_robot->clone();

        return simulated_robot;
    }

    void add_extra_to_simu(base_t::robot_simu_t& simu, const blackdrops::RolloutInfo& rollout_info) const
    {
        // std::cout << "Goal is: " << global::goal.transpose() << std::endl;
        // Change gravity
        Eigen::VectorXd gravity(3);
        gravity << 0., -9.81, 0.;
        simu.world()->setGravity(gravity);
        // Add goal marker
        Eigen::Vector6d goal_pose = Eigen::Vector6d::Zero();
        Eigen::VectorXd goal(3);
        goal << rollout_info.target(0), global::goal(1), rollout_info.target(1);
        goal_pose.tail(3) = goal;
        // pose, dims, type, mass, color, name
        simu.add_ellipsoid(goal_pose, {0.025, 0.025, 0.025}, "fixed", 1., dart::Color::Green(1.0), "goal_marker");
        // remove collisions from goal marker
        simu.world()->getSkeleton("goal_marker")->getRootBodyNode()->setCollidable(false);

        auto ground = global::global_floor->clone();
        Eigen::Vector6d floor_pose = Eigen::Vector6d::Zero();
        floor_pose(4) = -0.0125;
        simu.add_skeleton(ground, floor_pose, "fixed");

#ifdef GRAPHIC
        Eigen::Vector3d camera_pos = Eigen::Vector3d(0., 3., 0.);
        Eigen::Vector3d look_at = Eigen::Vector3d(0., 0., 0.);
        Eigen::Vector3d up = Eigen::Vector3d(0., 1., 0.);
        simu.graphics()->fixed_camera(camera_pos, look_at, up);
        // slow down visualization because 0.5 seconds is too fast
        simu.graphics()->set_render_period(0.03);
#endif
    }
};

struct RewardFunction : public blackdrops::reward::GPReward<RewardFunction, blackdrops::RewardGP<RewardParams>> {
    template <typename RolloutInfo>
    double operator()(const RolloutInfo& info, const Eigen::VectorXd& from_state, const Eigen::VectorXd& action, const Eigen::VectorXd& to_state, bool certain = false) const
    {
        Eigen::VectorXd goal = info.target;

        Eigen::VectorXd links(2);
        links << 0.1, 0.11;
        Eigen::VectorXd eef = tip(to_state.tail(2), links);

        double dee = (eef - goal).norm();
        double a = (action.array() / 200.0).square().sum();

        return -dee - a;
    }

    Eigen::VectorXd tip(const Eigen::VectorXd& theta, const Eigen::VectorXd& links) const
    {
        Eigen::VectorXd eef(2);

        eef(0) = links(0) * std::cos(theta(0));
        eef(1) = links(0) * std::sin(-theta(0));

        for (int i = 1; i < theta.size(); i++) {
            double th = 0.;
            for (int j = 0; j <= i; j++) {
                th += theta[j];
            }

            eef(0) += links(i) * std::cos(th);
            eef(1) += links(i) * std::sin(-th);
        }

        return eef;
    }

    template <typename RolloutInfo>
    Eigen::VectorXd get_sample(const RolloutInfo& info, const Eigen::VectorXd& from_state, const Eigen::VectorXd& action, const Eigen::VectorXd& to_state) const
    {
        Eigen::VectorXd vec(info.target.size() + 4);
        vec.head(2) = info.target;
        vec.segment(2, 2) = action;
        vec.tail(2) = to_state.tail(2);
        return vec;
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

    // std::cout << "Goal is: " << global::goal.transpose() << std::endl;
}

BO_DECLARE_DYN_PARAM(int, PolicyParams::nn_policy, hidden_neurons);
BO_DECLARE_DYN_PARAM(int, PolicyParams::gp_policy, pseudo_samples);
BO_DECLARE_DYN_PARAM(double, Params::blackdrops, boundary);
BO_DECLARE_DYN_PARAM(bool, Params::blackdrops, verbose);
BO_DECLARE_DYN_PARAM(bool, Params::blackdrops, stochastic);
BO_DECLARE_DYN_PARAM(int, Params::blackdrops, opt_evals);

BO_DECLARE_DYN_PARAM(double, Params::opt_cmaes, max_fun_evals);
BO_DECLARE_DYN_PARAM(double, Params::opt_cmaes, fun_tolerance);
BO_DECLARE_DYN_PARAM(double, Params::opt_cmaes, lbound);
BO_DECLARE_DYN_PARAM(double, Params::opt_cmaes, ubound);
BO_DECLARE_DYN_PARAM(int, Params::opt_cmaes, restarts);
BO_DECLARE_DYN_PARAM(int, Params::opt_cmaes, elitism);
BO_DECLARE_DYN_PARAM(bool, Params::opt_cmaes, handle_uncertainty);

class ReacherArgs : public utils::CmdArgs {
public:
    ReacherArgs() : utils::CmdArgs()
    {
        // clang-format off
        this->_desc.add_options()
                    ("opt_evals,o", po::value<int>(), "Number of rollouts for policy evaluation. Defaults to 10.");
        // clang-format on
    }

    int parse(int argc, char** argv)
    {
        int ret = utils::CmdArgs::parse(argc, argv);
        if (ret >= 0)
            return ret;

        try {
            po::variables_map vm;
            po::store(po::parse_command_line(argc, argv, this->_desc), vm);

            po::notify(vm);

            if (vm.count("opt_evals")) {
                int pl = vm["opt_evals"].as<int>();
                if (pl <= 0)
                    pl = 1;
                Params::blackdrops::set_opt_evals(pl);
            }
            else {
                Params::blackdrops::set_opt_evals(10);
            }
        }
        catch (po::error& e) {
            std::cerr << "[Exception caught while parsing command line arguments]: " << e.what() << std::endl;
            return 1;
        }

        return -1;
    }
};

int main(int argc, char** argv)
{
    ReacherArgs cmd_arguments;
    int ret = cmd_arguments.parse(argc, argv);
    if (ret >= 0)
        return ret;

    PolicyParams::nn_policy::set_hidden_neurons(cmd_arguments.neurons());
    PolicyParams::gp_policy::set_pseudo_samples(cmd_arguments.pseudo_samples());

    Params::blackdrops::set_boundary(cmd_arguments.boundary());
    Params::opt_cmaes::set_lbound(-cmd_arguments.boundary());
    Params::opt_cmaes::set_ubound(cmd_arguments.boundary());

    Params::opt_cmaes::set_max_fun_evals(cmd_arguments.max_fun_evals());
    Params::opt_cmaes::set_fun_tolerance(cmd_arguments.fun_tolerance());
    Params::opt_cmaes::set_restarts(cmd_arguments.restarts());
    Params::opt_cmaes::set_elitism(cmd_arguments.elitism());

#ifdef USE_TBB
    static tbb::task_scheduler_init init(cmd_arguments.threads());
#endif

    Params::blackdrops::set_verbose(cmd_arguments.verbose());
    Params::blackdrops::set_stochastic(cmd_arguments.stochastic());
    Params::opt_cmaes::set_handle_uncertainty(cmd_arguments.uncertainty());

    std::cout << std::endl;
    std::cout << "Cmaes parameters:" << std::endl;
    std::cout << "  max_fun_evals = " << Params::opt_cmaes::max_fun_evals() << std::endl;
    std::cout << "  fun_tolerance = " << Params::opt_cmaes::fun_tolerance() << std::endl;
    std::cout << "  restarts = " << Params::opt_cmaes::restarts() << std::endl;
    std::cout << "  elitism = " << Params::opt_cmaes::elitism() << std::endl;
    std::cout << "  handle_uncertainty = " << Params::opt_cmaes::handle_uncertainty() << std::endl;
    std::cout << "  stochastic rollouts = " << Params::blackdrops::stochastic() << std::endl;
    std::cout << "  parallel rollouts = " << Params::blackdrops::opt_evals() << std::endl;
    std::cout << "  boundary = " << Params::blackdrops::boundary() << std::endl;
    std::cout << "  tbb threads = " << cmd_arguments.threads() << std::endl;
    std::cout << std::endl;
    std::cout << "Policy parameters:" << std::endl;
#ifndef GPPOLICY
    std::cout << "  Type: Neural Network with 1 hidden layer and " << PolicyParams::nn_policy::hidden_neurons() << " hidden neurons." << std::endl;
#else
    std::cout << "  Type: Gaussian Process with " << PolicyParams::gp_policy::pseudo_samples() << " pseudo samples." << std::endl;
#endif
    std::cout << std::endl;

    init_simu(std::string(RESPATH) + "/skel/reacher2d.skel");

    using policy_opt_t = limbo::opt::Cmaes<Params>;

    using kernel_t = limbo::kernel::SquaredExpARD<Params>;
    using mean_t = limbo::mean::Constant<Params>;

    using GP_t = blackdrops::model::MultiGP<Params, limbo::model::GP, kernel_t, mean_t, blackdrops::model::multi_gp::MultiGPParallelLFOpt<Params, limbo::model::gp::KernelLFOpt<Params, limbo::opt::Rprop<Params>>>>;

    using MGP_t = blackdrops::GPModel<Params, GP_t>;

    blackdrops::BlackDROPS<Params, MGP_t, DARTReacher, global::policy_t, policy_opt_t, RewardFunction> reacher_system;

    reacher_system.learn(2, 20, true);
    // reacher_system.learn(1, 1, true);

    // ActualReward actual_reward;
    // std::ofstream ofs("reward_points.dat");
    // for (size_t i = 0; i < global::reward_gp.samples().size(); i++) {
    //     Eigen::VectorXd to_state = global::reward_gp.samples()[i];
    //     for (int j = 0; j < to_state.size(); j++)
    //         ofs << to_state[j] << " ";
    //     Eigen::VectorXd mu = global::reward_gp.mu(to_state);
    //     double r = actual_reward(to_state, to_state, to_state);
    //     ofs << mu(0) << " " << r;
    //     ofs << std::endl;
    // }
    // ofs.close();

    return 0;
}