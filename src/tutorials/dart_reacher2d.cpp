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
#include <limbo/kernel/squared_exp_ard.hpp>
#include <limbo/mean/constant.hpp>
#include <limbo/model/gp.hpp>
#include <limbo/model/multi_gp.hpp>
#include <limbo/model/multi_gp/parallel_lf_opt.hpp>
#include <limbo/opt/cmaes.hpp>

#include <blackdrops/blackdrops.hpp>
#include <blackdrops/model/gp/kernel_lf_opt.hpp>
#include <blackdrops/model/gp_model.hpp>
#include <blackdrops/system/dart_system.hpp>

#include <blackdrops/policy/nn_policy.hpp>

#include <blackdrops/reward/gp_reward.hpp>

#include <blackdrops/utils/cmd_args.hpp>
#include <blackdrops/utils/dart_utils.hpp>
#include <blackdrops/utils/utils.hpp>

struct Params {
    struct blackdrops : public ::blackdrops::defaults::blackdrops {
        BO_PARAM(size_t, action_dim, 2);
        BO_PARAM(size_t, model_input_dim, 6);
        BO_PARAM(size_t, model_pred_dim, 4);
        // BO_PARAM(double, dt, 0.1);
        // BO_PARAM(double, T, 3.0);
        BO_PARAM(double, dt, 0.02);
        BO_PARAM(double, T, 0.5);
        BO_DYN_PARAM(double, boundary);
        BO_DYN_PARAM(bool, verbose);
        BO_DYN_PARAM(bool, stochastic);
    };

    struct dart_system {
        BO_PARAM(double, sim_step, 0.001);
    };

    struct dart_policy_control {
        // BO_PARAM_STRING(joint_type, "servo");
        BO_PARAM_STRING(joint_type, "torque");
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
        BO_DYN_PARAM(int, max_fun_evals);
        BO_DYN_PARAM(double, fun_tolerance);
        BO_DYN_PARAM(int, restarts);
        BO_DYN_PARAM(int, elitism);
        BO_DYN_PARAM(bool, handle_uncertainty);

        BO_DYN_PARAM(int, lambda);

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

struct RewardParams : public blackdrops::reward_defaults {
    struct mean_constant {
        BO_PARAM(double, constant, -1.);
    };
};

namespace global {
    std::shared_ptr<robot_dart::Robot> global_robot;
    dart::dynamics::SkeletonPtr global_floor;

    using policy_t = blackdrops::policy::NNPolicy<PolicyParams>;

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
    PolicyControl(const Eigen::VectorXd& ctrl) : base_t(ctrl) {}

    Eigen::VectorXd get_state(const base_t::robot_t& robot) const
    {
        return get_robot_state(robot);
    }

    std::shared_ptr<robot_dart::control::RobotControl> clone() const override
    {
        return std::make_shared<PolicyControl>(*this);
    }
};

struct DARTReacher : public blackdrops::system::DARTSystem<Params, PolicyControl, blackdrops::RolloutInfo> {
    using base_t = blackdrops::system::DARTSystem<Params, PolicyControl, blackdrops::RolloutInfo>;

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

    void add_extra_to_simu(base_t::robot_simu_t& simu, const blackdrops::RolloutInfo& info) const
    {
        // Change gravity
        Eigen::VectorXd gravity(3);
        gravity << 0., -9.81, 0.;
        simu.world()->setGravity(gravity);
        // Add goal marker
        Eigen::Vector6d goal_pose = Eigen::Vector6d::Zero();
        goal_pose.tail(3) = global::goal;
        // dims, pose, type, mass, color, name
        auto ellipsoid = robot_dart::Robot::create_ellipsoid({0.025, 0.025, 0.025}, goal_pose, "fixed", 1., dart::Color::Green(1.0), "goal_marker");
        // remove collisions from goal marker
        ellipsoid->skeleton()->getRootBodyNode()->setCollidable(false);
        // add ellipsoid to simu
        simu.add_robot(ellipsoid);

        auto ground = global::global_floor->clone();
        Eigen::Vector6d floor_pose = Eigen::Vector6d::Zero();
        floor_pose(4) = -0.0125;
        auto floor_robot = std::make_shared<robot_dart::Robot>(ground, "ground");
        floor_robot->skeleton()->setPositions(floor_pose);
        floor_robot->fix_to_world();
        // add floor to simu
        simu.add_robot(floor_robot);

#ifdef GRAPHIC
        // Add 2 directional lights without shadows
        auto graphics = std::static_pointer_cast<robot_dart::gui::magnum::Graphics>(simu.graphics());
        graphics->clear_lights();
        robot_dart::gui::magnum::gs::Material mat;
        mat.diffuse_color() = {1.f, 1.f, 1.f, 1.f};
        mat.specular_color() = {1.f, 1.f, 1.f, 1.f};
        Magnum::Vector3 dir = {-0.5f, -0.5f, -0.5f};
        robot_dart::gui::magnum::gs::Light light = robot_dart::gui::magnum::gs::create_directional_light(dir, mat);
        light.set_casts_shadows(false);
        graphics->add_light(light);
        dir = {0.5f, -0.5f, 0.5f};
        light = robot_dart::gui::magnum::gs::create_directional_light(dir, mat);
        light.set_casts_shadows(false);
        graphics->add_light(light);
#endif
    }

    void set_robot_state(const std::shared_ptr<robot_dart::Robot>& robot, const Eigen::VectorXd& state) const
    {
        robot->skeleton()->setVelocities(state.head(2));
        robot->skeleton()->setPositions(state.tail(2));
    }
};

struct RewardFunction : public blackdrops::reward::GPReward<RewardFunction, blackdrops::RewardGP<RewardParams>> {
    template <typename RolloutInfo>
    double operator()(const RolloutInfo& info, const Eigen::VectorXd& from_state, const Eigen::VectorXd& action, const Eigen::VectorXd& to_state, bool certain = false) const
    {
        Eigen::VectorXd goal(2);
        goal << global::goal(0), global::goal(2);

        Eigen::VectorXd links(2);
        links << 0.1, 0.11;
        Eigen::VectorXd eef = tip(to_state.tail(2), links);

        double dee = (eef - goal).norm();

        return -dee;
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
        return to_state.tail(2);
    }
};

void init_simu(const std::string& robot_file)
{
    global::global_robot = std::make_shared<robot_dart::Robot>(blackdrops::utils::load_skel(robot_file, "arm"));
    Eigen::Isometry3d tf = global::global_robot->skeleton()->getRootBodyNode()->getParentJoint()->getTransformFromParentBodyNode();
    tf.translation() = Eigen::Vector3d(0., 0.01, 0.);
    global::global_robot->skeleton()->getRootBodyNode()->getParentJoint()->setTransformFromParentBodyNode(tf);

    global::global_floor = blackdrops::utils::load_skel(robot_file, "ground skeleton");

    global::goal = Eigen::Vector3d(0.1, 0.01, -0.1);

    std::cout << "Goal is: " << global::goal.transpose() << std::endl;
}

BO_DECLARE_DYN_PARAM(int, PolicyParams::nn_policy, hidden_neurons);
BO_DECLARE_DYN_PARAM(double, Params::blackdrops, boundary);
BO_DECLARE_DYN_PARAM(bool, Params::blackdrops, verbose);
BO_DECLARE_DYN_PARAM(bool, Params::blackdrops, stochastic);

BO_DECLARE_DYN_PARAM(int, Params::opt_cmaes, max_fun_evals);
BO_DECLARE_DYN_PARAM(double, Params::opt_cmaes, fun_tolerance);
BO_DECLARE_DYN_PARAM(double, Params::opt_cmaes, lbound);
BO_DECLARE_DYN_PARAM(double, Params::opt_cmaes, ubound);
BO_DECLARE_DYN_PARAM(int, Params::opt_cmaes, restarts);
BO_DECLARE_DYN_PARAM(int, Params::opt_cmaes, elitism);
BO_DECLARE_DYN_PARAM(int, Params::opt_cmaes, lambda);
BO_DECLARE_DYN_PARAM(bool, Params::opt_cmaes, handle_uncertainty);

int main(int argc, char** argv)
{
    blackdrops::utils::CmdArgs cmd_arguments;
    int ret = cmd_arguments.parse(argc, argv);
    if (ret >= 0)
        return ret;

    PolicyParams::nn_policy::set_hidden_neurons(cmd_arguments.neurons());
    // PolicyParams::gp_policy::set_pseudo_samples(cmd_arguments.pseudo_samples());

    Params::blackdrops::set_boundary(cmd_arguments.boundary());
    Params::opt_cmaes::set_lbound(-cmd_arguments.boundary());
    Params::opt_cmaes::set_ubound(cmd_arguments.boundary());

    Params::opt_cmaes::set_max_fun_evals(cmd_arguments.max_fun_evals());
    Params::opt_cmaes::set_fun_tolerance(cmd_arguments.fun_tolerance());
    Params::opt_cmaes::set_restarts(cmd_arguments.restarts());
    Params::opt_cmaes::set_elitism(cmd_arguments.elitism());
    Params::opt_cmaes::set_lambda(cmd_arguments.lambda());

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
    std::cout << "  boundary = " << Params::blackdrops::boundary() << std::endl;
    std::cout << "  tbb threads = " << cmd_arguments.threads() << std::endl;
    std::cout << std::endl;
    std::cout << "Policy parameters:" << std::endl;
    std::cout << "  Type: Neural Network with 1 hidden layer and " << PolicyParams::nn_policy::hidden_neurons() << " hidden neurons." << std::endl;
    std::cout << std::endl;

    init_simu(std::string(RESPATH) + "/skel/reacher2d.skel");

    using policy_opt_t = limbo::opt::Cmaes<Params>;

    using kernel_t = limbo::kernel::SquaredExpARD<Params>;
    using mean_t = limbo::mean::Constant<Params>;

    using GP_t = limbo::model::MultiGP<Params, limbo::model::GP, kernel_t, mean_t, limbo::model::multi_gp::ParallelLFOpt<Params, blackdrops::model::gp::KernelLFOpt<Params>>>;

    using MGP_t = blackdrops::model::GPModel<Params, GP_t>;

    blackdrops::BlackDROPS<Params, MGP_t, DARTReacher, global::policy_t, policy_opt_t, RewardFunction> reacher_system;

    reacher_system.learn(2, 15, true);

    return 0;
}