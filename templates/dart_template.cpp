//| Copyright Inria July 2017
//| This project has received funding from the European Research Council (ERC) under
//| the European Union's Horizon 2020 research and innovation programme (grant
//| agreement No 637972) - see http://www.resibots.eu
//|
//| Contributor(s):
//|   - Konstantinos Chatzilygeroudis (konstantinos.chatzilygeroudis@inria.fr)
//|   - Roberto Rama (bertoski@gmail.com)
//|
//| This software is a computer library whose purpose is to optimize continuous,
//| black-box functions. It mainly implements Gaussian processes and Bayesian
//| optimization.
//| Main repository: http://github.com/resibots/blackdrops
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

// TO-CHANGE (optional): You can include other policies as well (GP and linear policy already implemented)
#include <blackdrops/policy/nn_policy.hpp>

#include <utils/dart_utils.hpp>
#include <utils/utils.hpp>

struct Params {
#ifdef GRAPHIC
    struct graphics : robot_dart::defaults::graphics {
    };
#endif

    struct blackdrops {
        // TO-CHANGE: Here you should set your parameters
        BO_PARAM(size_t, action_dim, @int_value); // action space # of dimensions
        BO_PARAM(size_t, model_input_dim, @int_value); // transformed input # of dimensions (input to the GPs and policy)
        BO_PARAM(size_t, model_pred_dim, @int_value); // state space # of dimensions
        BO_PARAM(double, dt, @double_value); // sampling/control rate
        BO_PARAM(double, T, @double_value); // duration of each episode
        BO_DYN_PARAM(double, boundary);
        BO_DYN_PARAM(bool, verbose);
    };

    struct dart_system {
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
        // TO-CHANGE: fill in the appropriate values
        BO_PARAM_ARRAY(double, max_u, @double_values_separated_by_comma); // max (absolute) values for the actions (one per dimension)
        BO_PARAM_ARRAY(double, limits, @double_values_separated_by_comma); // normalization factor for the inputs of the neural network policy
        BO_DYN_PARAM(int, hidden_neurons);
        // TO-CHANGE (optional): if you want your NN policy to produce actions closer to 0, change the value of af to something close to zero (i.e., 0.2)
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

    // TO-CHANGE: If you want to use some other policy other than neural network, you should change this typedef
    using policy_t = blackdrops::policy::NNPolicy<PolicyParams>;

    // TO-CHANGE: You may want to define some global helper variables
}

struct PolicyControl : public blackdrops::system::BaseDARTPolicyControl<Params, global::policy_t> {
    using base_t = blackdrops::system::BaseDARTPolicyControl<Params, global::policy_t>;

    PolicyControl() : base_t() {}
    PolicyControl(const std::vector<double>& ctrl, base_t::robot_t robot) : base_t(ctrl, robot) {}

    Eigen::VectorXd get_state(const base_t::robot_t& robot, bool full) const
    {
        // TO-CHANGE: write code to get the state of your robot
        // if full equals true, then you should return the transformed state
    }
};

// TO-CHANGE: Change the MyDARTSystem to your desired name
struct MyDARTSystem : public blackdrops::system::DARTSystem<Params, PolicyControl> {
    using base_t = blackdrops::system::DARTSystem<Params, PolicyControl>;

    Eigen::VectorXd init_state() const
    {
        // TO-CHANGE: return my initial state
    }

    Eigen::VectorXd transform_state(const Eigen::VectorXd& original_state) const
    {
        // TO-CHANGE: Code to transform your state (for GP and policy input) if needed
        // if not needed, just return the original state
    }

    std::shared_ptr<robot_dart::Robot> get_robot() const
    {
        // TO-CHANGE: if you want to create your robot in a different way, you can change this part
        std::shared_ptr<robot_dart::Robot> simulated_robot = global::global_robot->clone();

        return simulated_robot;
    }

    void add_extra_to_simu(base_t::robot_simu_t& simu) const
    {
        // TO-CHANGE: if you want, you can add some extra to your simulator object (this is called once before its episode on a newly-created simulator object)
    }
};

struct RewardFunction {
    double operator()(const Eigen::VectorXd& from_state, const Eigen::VectorXd& action, const Eigen::VectorXd& to_state, bool certain = false) const
    {
        // TO-CHANGE: return the immediate reward function r(x,u,x')
    }
};

void init_simu(const std::string& robot_file)
{
    // TO-CHANGE: write code to load your robot
    // if you want to load a SKEL file, use std::make_shared<robot_dart::Robot>(utils::load_skel(robot_file, "name_of_the_skeleton"))
    // if you want to load a URDF or an SDF file, use std::make_shared<robot_dart::Robot>(robot_dart::Robot(robot_file, {}, "give_a_name_to_your_robot", true));

    // global::global_robot = ....;
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

    // TO-CHANGE: change @path_to_file to your file: /URDF/my_robot.urdf
    // or pass a different input to init_simu: "/home/myusername/path/to/myrobot.sdf"
    init_simu(std::string(RESPATH) + @path_to_file);

    using policy_opt_t = limbo::opt::Cmaes<Params>;

    using kernel_t = limbo::kernel::SquaredExpARD<Params>;
    using mean_t = limbo::mean::Constant<Params>;

    using GP_t = blackdrops::model::MultiGP<Params, limbo::model::GP, kernel_t, mean_t, blackdrops::model::multi_gp::MultiGPParallelLFOpt<Params, limbo::model::gp::KernelLFOpt<Params>>>;

    using MGP_t = blackdrops::GPModel<Params, GP_t>;

    // TO-CHANGE: Change the MyDARTSystem to your desired name
    blackdrops::BlackDROPS<Params, MGP_t, MyDARTSystem, global::policy_t, policy_opt_t, RewardFunction> my_system;

    my_system.learn(2, 15, true);

    return 0;
}