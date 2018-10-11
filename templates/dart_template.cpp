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
// #define MEAN // TO-CHANGE (optional): uncomment this if you want to use a mean function
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

// TO-CHANGE (optional): You can include other policies as well (GP and linear policy already implemented)
#include <blackdrops/policy/nn_policy.hpp>

// TO-CHANGE (optional): You can include other reward types as well (GPReward is already implemented)
#include <blackdrops/reward/reward.hpp>

#include <blackdrops/utils/cmd_args.hpp>
#include <blackdrops/utils/dart_utils.hpp>
#include <blackdrops/utils/utils.hpp>

struct Params {
    struct blackdrops : public ::blackdrops::defaults::blackdrops {
        // TO-CHANGE: Here you should set your parameters
        BO_PARAM(size_t, action_dim, @int_value); // action space # of dimensions
        BO_PARAM(size_t, model_input_dim, @int_value); // transformed input # of dimensions (input to the GPs and policy)
        BO_PARAM(size_t, model_pred_dim, @int_value); // state space # of dimensions
        BO_PARAM(double, dt, @double_value); // sampling/control rate
        BO_PARAM(double, T, @double_value); // duration of each episode
        BO_DYN_PARAM(bool, verbose);
        BO_DYN_PARAM(bool, stochastic);
        BO_DYN_PARAM(double, boundary);
    };

    struct dart_system {
        // TO-CHANGE: This is the simulation step for the DART simulation, you can set it to whatever you want (it should be smaller than `dt` above)
        BO_PARAM(double, sim_step, 0.001);
    };

    struct dart_policy_control {
        // TO-CHANGE: Select the type of actuator. Possible values: dart::dynamics::Joint::FORCE, dart::dynamics::Joint::SERVO, dart::dynamics::Joint::VELOCITY
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
        // TO-CHANGE: fill in the appropriate values
        BO_PARAM_ARRAY(double, max_u, @double_values_separated_by_comma); // max (absolute) values for the actions (one per dimension): [-max, max]
        BO_PARAM_ARRAY(double, limits, @double_values_separated_by_comma); // normalization factor for the inputs of the neural network policy
        BO_DYN_PARAM(int, hidden_neurons);
        // TO-CHANGE (optional): if you want your NN policy to produce actions closer to 0, change the value of af to something close to zero (i.e., 0.2)
        BO_PARAM(double, af, 1.0);
    };
};

namespace global {
    std::shared_ptr<robot_dart::Robot> global_robot;

    // TO-CHANGE: If you want to use some other policy other than neural network, you should change this typedef
    using policy_t = blackdrops::policy::NNPolicy<PolicyParams>;

    // TO-CHANGE: You may want to define some global helper variables
} // namespace global

struct PolicyControl : public blackdrops::system::BaseDARTPolicyControl<Params, global::policy_t> {
    using base_t = blackdrops::system::BaseDARTPolicyControl<Params, global::policy_t>;

    PolicyControl() : base_t() {}
    PolicyControl(const std::vector<double>& ctrl) : base_t(ctrl) {}

    Eigen::VectorXd get_state(const robot_t& robot) const
    {
        // TO-CHANGE: write code to get the state of your robot
    }

    std::shared_ptr<robot_dart::control::RobotControl> clone() const override
    {
        return std::make_shared<PolicyControl>(*this);
    }
};

// TO-CHANGE: Change the MyDARTSystem to your desired name
// TO-CHANGE: Change the third template argument to your desired RolloutInfo structure
struct MyDARTSystem : public blackdrops::system::DARTSystem<Params, PolicyControl, blackdrops::RolloutInfo> {
    using base_t = blackdrops::system::DARTSystem<Params, PolicyControl, blackdrops::RolloutInfo>;

    Eigen::VectorXd init_state() const
    {
        // TO-CHANGE: return my initial state
        // if you omit this function, the zero state is returned
    }

    Eigen::VectorXd transform_state(const Eigen::VectorXd& original_state) const
    {
        // TO-CHANGE: Code to transform your state (for GP and policy input) if needed
        // if not needed, just return the original state
        // if you omit this function, no transformation is applied
    }

    Eigen::VectorXd add_noise(const Eigen::VectorXd& original_state) const
    {
        // TO-CHANGE: Code to add observation noise to the system
        // you should return the full noisy state, not just the noise
        // if no noise is desired, just return the original state
        // if you omit this function, no noise is added
    }

    Eigen::VectorXd policy_transform(const Eigen::VectorXd& original_state, RolloutInfo* info) const
    {
        // TO-CHANGE: Code to transform the state variables that go to the policy if needed
        // the input original_state is the transformed state (by the transform_state variable)
    }

    std::shared_ptr<robot_dart::Robot> get_robot() const
    {
        // TO-CHANGE: if you want to create your robot in a different way, you can change this part
        std::shared_ptr<robot_dart::Robot> simulated_robot = global::global_robot->clone();

        return simulated_robot;
    }

    void add_extra_to_simu(base_t::robot_simu_t& simu, const blackdrops::RolloutInfo& info) const
    {
        // TO-CHANGE: if you want, you can add some extra to your simulator object (this is called once before its episode on a newly-created simulator object)
    }

    void set_robot_state(const std::shared_ptr<robot_dart::Robot>& robot, const Eigen::VectorXd& state) const
    {
        // TO-CHANGE: This is called whenever is needed to set the robot in a specific state
    }
};

struct RewardFunction : public blackdrops::reward::Reward<RewardFunction> {
    template <typename RolloutInfo>
    double operator()(const RolloutInfo& info, const Eigen::VectorXd& from_state, const Eigen::VectorXd& action, const Eigen::VectorXd& to_state) const
    {
        // TO-CHANGE: return the immediate reward function r(x,u,x')
        // the RolloutInfo can contain information needed for the computation of the reward function
    }
};

void init_simu(const std::string& robot_file)
{
    // TO-CHANGE: write code to load your robot
    // if you want to load a SKEL file, use std::make_shared<robot_dart::Robot>(utils::load_skel(robot_file, "name_of_the_skeleton"))
    // if you want to load a URDF or an SDF file, use std::make_shared<robot_dart::Robot>(robot_dart::Robot(robot_file, "give_a_name_to_your_robot"));

    // global::global_robot = ....;
}

BO_DECLARE_DYN_PARAM(int, PolicyParams::nn_policy, hidden_neurons);
BO_DECLARE_DYN_PARAM(double, Params::blackdrops, boundary);
BO_DECLARE_DYN_PARAM(bool, Params::blackdrops, verbose);
BO_DECLARE_DYN_PARAM(bool, Params::blackdrops, stochastic);
BO_DECLARE_DYN_PARAM(double, Params::blackdrops, boundary);

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
    utils::CmdArgs cmd_arguments;
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
    // TO-CHANGE: Here you can log your policy if you are not using a neural network
    std::cout << "  Type: Neural Network with 1 hidden layer and " << PolicyParams::nn_policy::hidden_neurons() << " hidden neurons." << std::endl;
    std::cout << std::endl;

    // TO-CHANGE: change @path_to_file to your file: /URDF/my_robot.urdf
    // or pass a different input to init_simu: "/home/myusername/path/to/myrobot.sdf"
    init_simu(std::string(RESPATH) + @path_to_file);

    using policy_opt_t = limbo::opt::Cmaes<Params>;

    using kernel_t = limbo::kernel::SquaredExpARD<Params>;
    using mean_t = limbo::mean::Constant<Params>;

    using GP_t = limbo::model::MultiGP<Params, limbo::model::GP, kernel_t, mean_t, limbo::model::multi_gp::ParallelLFOpt<Params, blackdrops::model::gp::KernelLFOpt<Params>>>;

    using MGP_t = blackdrops::model::GPModel<Params, GP_t>;

    // TO-CHANGE: Change the MyDARTSystem to your desired name
    blackdrops::BlackDROPS<Params, MGP_t, MyDARTSystem, global::policy_t, policy_opt_t, RewardFunction> my_system;

    // TO-CHANGE: fill in the data
    my_system.learn(@initial_random_trials, @learning_episodes, @random_policies, [@policy_file]); // @random_policies -- this should be true if you want to always start the policy optimization from the best so far tried policy (if false, the optimization will start from the previous policy tried on the robot)
    // @policy_file is an optional argument that if you are using a mean function, you can set it to the path to an initial policy to try

    return 0;
}