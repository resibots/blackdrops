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
#include <blackdrops/gp_model.hpp>
#include <blackdrops/model/gp/kernel_lf_opt.hpp>
#include <blackdrops/system/dart_system.hpp>

#include <blackdrops/policy/nn_policy.hpp>

#include <blackdrops/reward/gp_reward.hpp>

#include <blackdrops/utils/cmd_args.hpp>
#include <blackdrops/utils/utils.hpp>

struct Params {
    struct blackdrops : public ::blackdrops::defaults::blackdrops {
        BO_PARAM(size_t, action_dim, 4);
        BO_PARAM(size_t, model_input_dim, 8);
        BO_PARAM(size_t, model_pred_dim, 4);
        BO_PARAM(double, dt, 0.1);
        BO_PARAM(double, T, 4.0);
        BO_DYN_PARAM(double, boundary);
        BO_DYN_PARAM(bool, verbose);
        BO_DYN_PARAM(bool, stochastic);
    };

    struct dart_system {
        BO_PARAM(double, sim_step, 0.001);
    };

    struct dart_policy_control {
        BO_PARAM(dart::dynamics::Joint::ActuatorType, joint_type, dart::dynamics::Joint::VELOCITY);
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
        // Velocity limits
        // BO_PARAM_ARRAY(double, max_u, 3.0, 3.0, 3.0, 3.0);
        BO_PARAM_ARRAY(double, max_u, 1.0, 1.0, 1.0, 1.0);
        BO_PARAM_ARRAY(double, limits, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
        BO_DYN_PARAM(int, hidden_neurons);
        BO_PARAM(double, af, 1.0);
    };
};

namespace global {
    std::shared_ptr<robot_dart::Robot> global_robot;

    using policy_t = blackdrops::policy::NNPolicy<PolicyParams>;

    Eigen::VectorXd goal(3);
} // namespace global

Eigen::VectorXd get_robot_state(const std::shared_ptr<robot_dart::Robot>& robot)
{
    Eigen::VectorXd state = robot->skeleton()->getPositions();

    return state;
}

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

struct PolicyControl : public blackdrops::system::BaseDARTPolicyControl<Params, global::policy_t> {
    using base_t = blackdrops::system::BaseDARTPolicyControl<Params, global::policy_t>;

    PolicyControl() : base_t() {}
    PolicyControl(const std::vector<double>& ctrl) : base_t(ctrl) {}

    Eigen::VectorXd get_state(const robot_t& robot) const
    {
        return get_robot_state(robot);
    }

    std::shared_ptr<robot_dart::control::RobotControl> clone() const override
    {
        return std::make_shared<PolicyControl>(*this);
    }
};

struct SimpleArm : public blackdrops::system::DARTSystem<Params, PolicyControl, blackdrops::RolloutInfo> {
    using base_t = blackdrops::system::DARTSystem<Params, PolicyControl, blackdrops::RolloutInfo>;

    Eigen::VectorXd init_state() const
    {
        return Eigen::VectorXd::Zero(Params::blackdrops::model_pred_dim());
    }

    Eigen::VectorXd transform_state(const Eigen::VectorXd& original_state) const
    {
        Eigen::VectorXd ret = Eigen::VectorXd::Zero(Params::blackdrops::model_input_dim());
        for (int j = 0; j < original_state.size(); j++) {
            ret(2 * j) = std::cos(original_state(j));
            ret(2 * j + 1) = std::sin(original_state(j));
        }

        return ret;
    }

    std::shared_ptr<robot_dart::Robot> get_robot() const
    {
        std::shared_ptr<robot_dart::Robot> simulated_robot = global::global_robot->clone();
        simulated_robot->fix_to_world();
        simulated_robot->set_position_enforced(true);

        return simulated_robot;
    }

    void add_extra_to_simu(base_t::robot_simu_t& simu, const blackdrops::RolloutInfo& info) const
    {
        // Add goal marker
        Eigen::Vector6d goal_pose = Eigen::Vector6d::Zero();
        goal_pose.tail(3) = global::goal;
        // dims, pose, type, mass, color, name
        auto ellipsoid = robot_dart::Robot::create_ellipsoid({0.1, 0.1, 0.1}, goal_pose, "fixed", 1., dart::Color::Green(1.0), "goal_marker");
        // remove collisions from goal marker
        ellipsoid->skeleton()->getRootBodyNode()->setCollidable(false);
        // add ellipsoid to simu
        simu.add_robot(ellipsoid);
    }
};

struct RewardFunction : public blackdrops::reward::GPReward<RewardFunction> {
    template <typename RolloutInfo>
    double operator()(const RolloutInfo& info, const Eigen::VectorXd& from_state, const Eigen::VectorXd& action, const Eigen::VectorXd& to_state) const
    {
        std::shared_ptr<robot_dart::Robot> simulated_robot = global::global_robot->clone();
        simulated_robot->fix_to_world();
        simulated_robot->set_position_enforced(true);

        simulated_robot->skeleton()->setPositions(to_state);

        auto bd = simulated_robot->skeleton()->getBodyNode("arm_link_5");
        Eigen::VectorXd eef = bd->getTransform().translation();
        double s_c_sq = 0.2 * 0.2;
        double dee = (eef - global::goal).squaredNorm();

        return std::exp(-0.5 / s_c_sq * dee);
    }

    template <typename RolloutInfo>
    Eigen::VectorXd get_sample(const RolloutInfo& info, const Eigen::VectorXd& from_state, const Eigen::VectorXd& action, const Eigen::VectorXd& to_state) const
    {
        return to_state;
    }
};

void init_simu(const std::string& robot_file)
{
    global::global_robot = std::make_shared<robot_dart::Robot>(robot_file, "arm");

    // get goal position
    std::shared_ptr<robot_dart::Robot> simulated_robot = global::global_robot->clone();
    simulated_robot->fix_to_world();
    simulated_robot->set_position_enforced(true);
    // here set goal position
    Eigen::VectorXd positions(4);
    positions << M_PI / 4.0, M_PI / 8.0, M_PI / 8.0, M_PI / 8.0;
    simulated_robot->skeleton()->setPositions(positions);

    auto bd = simulated_robot->skeleton()->getBodyNode("arm_link_5");
    global::goal = bd->getTransform().translation();

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

    init_simu(std::string(RESPATH) + "/URDF/arm.urdf");

    using policy_opt_t = limbo::opt::Cmaes<Params>;

    using kernel_t = limbo::kernel::SquaredExpARD<Params>;
    using mean_t = limbo::mean::Constant<Params>;

    using GP_t = limbo::model::MultiGP<Params, limbo::model::GP, kernel_t, mean_t, limbo::model::multi_gp::ParallelLFOpt<Params, blackdrops::model::gp::KernelLFOpt<Params>>>;

    using MGP_t = blackdrops::GPModel<Params, GP_t>;

    blackdrops::BlackDROPS<Params, MGP_t, SimpleArm, blackdrops::policy::NNPolicy<PolicyParams>, policy_opt_t, RewardFunction> arm_system;

    arm_system.learn(1, 15, true);

    return 0;
}