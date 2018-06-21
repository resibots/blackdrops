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
#include <limbo/opt/cmaes.hpp>
#include <limbo/opt/rprop.hpp>

#include <blackdrops/blackdrops.hpp>
#include <blackdrops/gp_model.hpp>
#include <blackdrops/gp_multi_model.hpp>
#include <blackdrops/model/gp/kernel_lf_opt.hpp>
#include <blackdrops/model/multi_gp.hpp>
#include <blackdrops/model/multi_gp/multi_gp_parallel_opt.hpp>
#include <blackdrops/system/ode_system.hpp>

#include <blackdrops/policy/nn_policy.hpp>

#include <blackdrops/reward/reward.hpp>

#include <utils/cmd_args.hpp>
#include <utils/utils.hpp>

#if defined(USE_SDL) && !defined(NODSP)
#include <SDL2/SDL.h>

//Screen dimension constants
const int SCREEN_WIDTH = 640;
const int SCREEN_HEIGHT = 480;

//The window we'll be rendering to
SDL_Window* window = NULL;

//The window renderer
SDL_Renderer* renderer = NULL;

bool sdl_init()
{
    //Initialize SDL
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        std::cout << "SDL could not initialize! SDL_Error: " << SDL_GetError() << std::endl;
        return false;
    }

    window = SDL_CreateWindow("Cartpole Task", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, SCREEN_WIDTH, SCREEN_HEIGHT, SDL_WINDOW_SHOWN);
    if (window == NULL) {
        std::cout << "Window could not be created! SDL_Error: " << SDL_GetError() << std::endl;
        return false;
    }

    //Create renderer for window
    renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_SOFTWARE);
    if (renderer == NULL) {
        std::cout << "Renderer could not be created! SDL Error: " << SDL_GetError() << std::endl;
        return false;
    }

    //Initialize renderer color
    SDL_SetRenderDrawColor(renderer, 0xFF, 0xFF, 0xFF, 0xFF);

    //Update the surface
    SDL_UpdateWindowSurface(window);

    //If everything initialized fine
    return true;
}

bool draw_cartpole(double x, double theta, bool red = false)
{
    double th_x = std::cos(theta), th_y = std::sin(theta);

    SDL_Rect outlineRect = {static_cast<int>(SCREEN_WIDTH / 2 - x * SCREEN_HEIGHT / 4 - 0.1 * SCREEN_HEIGHT / 4), static_cast<int>(SCREEN_HEIGHT / 2 - 0.05 * SCREEN_HEIGHT / 4), static_cast<int>(0.2 * SCREEN_HEIGHT / 4), static_cast<int>(0.1 * SCREEN_HEIGHT / 4)};
    SDL_SetRenderDrawColor(renderer, 0x00, 0x00, 0xFF, 0xFF);
    if (red)
        SDL_SetRenderDrawColor(renderer, 0xFF, 0x00, 0x00, 0xFF);
    SDL_RenderFillRect(renderer, &outlineRect);

    SDL_RenderDrawLine(renderer, SCREEN_WIDTH / 2 - x * SCREEN_HEIGHT / 4, SCREEN_HEIGHT / 2, SCREEN_WIDTH / 2 - x * SCREEN_HEIGHT / 4 + th_y * SCREEN_HEIGHT / 8, SCREEN_HEIGHT / 2 + th_x * SCREEN_HEIGHT / 8);

    return true;
}

bool draw_goal(double x, double y)
{
    SDL_Rect outlineRect = {static_cast<int>(SCREEN_WIDTH / 2 + x * SCREEN_WIDTH / 4 - 0.05 * SCREEN_HEIGHT / 4), static_cast<int>(SCREEN_HEIGHT / 2 - y * SCREEN_HEIGHT / 4 - 0.05 * SCREEN_HEIGHT / 4), static_cast<int>(0.1 * SCREEN_HEIGHT / 4), static_cast<int>(0.1 * SCREEN_HEIGHT / 4)};
    SDL_SetRenderDrawColor(renderer, 0xFF, 0x00, 0x00, 0xFF);
    SDL_RenderFillRect(renderer, &outlineRect);

    return true;
}

void sdl_clean()
{
    // Destroy renderer
    SDL_DestroyRenderer(renderer);
    //Destroy window
    SDL_DestroyWindow(window);
    //Quit SDL
    SDL_Quit();
}
#endif

struct Params {
    BO_PARAM(double, goal_pos, M_PI);
    BO_PARAM(double, goal_vel, 0.0);
    BO_PARAM(double, goal_pos_x, 0.0);
    BO_PARAM(double, goal_vel_x, 0.0);

    struct blackdrops : public ::blackdrops::defaults::blackdrops {
        BO_PARAM(size_t, action_dim, 1);
        BO_PARAM(size_t, model_input_dim, 5);
        BO_PARAM(size_t, model_pred_dim, 4);
        BO_PARAM(double, dt, 0.1);
        BO_PARAM(double, T, 4.0);
        BO_DYN_PARAM(double, boundary);
        BO_DYN_PARAM(bool, verbose);
        BO_DYN_PARAM(bool, stochastic);

        BO_PARAM(bool, stochastic_evaluation, true);
        BO_PARAM(int, num_evals, 500);
        // BO_PARAM(int, opt_evals, 5);
        BO_DYN_PARAM(int, opt_evals);
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

    struct opt_rprop : public limbo::defaults::opt_rprop {
        BO_PARAM(int, iterations, 300);
        BO_PARAM(double, eps_stop, 1e-4);
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
};

struct PolicyParams {
    struct blackdrops : public Params::blackdrops {
    };

    struct nn_policy {
        BO_PARAM(size_t, state_dim, Params::blackdrops::model_input_dim());
        BO_PARAM(size_t, action_dim, Params::blackdrops::action_dim());
        BO_PARAM_ARRAY(double, max_u, 10.0);
        BO_DYN_PARAM(int, hidden_neurons);
        BO_PARAM_ARRAY(double, limits, 5., 5., 10., 1., 1.);
        BO_PARAM(double, af, 1.0);
    };
};

struct CartPole : public blackdrops::system::ODESystem<Params, blackdrops::RolloutInfo> {
    Eigen::VectorXd init_state() const
    {
        constexpr double sigma = 0.001;

        Eigen::VectorXd st = gaussian_rand(Eigen::VectorXd::Zero(4), sigma);

        return st;
    }

    Eigen::VectorXd transform_state(const Eigen::VectorXd& original_state) const
    {
        Eigen::VectorXd trans_state = Eigen::VectorXd::Zero(5);
        trans_state.head(3) = original_state.head(3);
        trans_state(3) = std::cos(original_state(3));
        trans_state(4) = std::sin(original_state(3));

        return trans_state;
    }

    Eigen::VectorXd add_noise(const Eigen::VectorXd& original_state) const
    {
        constexpr double sigma = 0.01;

        Eigen::VectorXd noisy = gaussian_rand(original_state, sigma);

        return noisy;
    }

    void draw_single(const Eigen::VectorXd& state) const
    {
#if defined(USE_SDL) && !defined(NODSP)
        double dt = Params::blackdrops::dt();
        //Clear screen
        SDL_SetRenderDrawColor(renderer, 0xFF, 0xFF, 0xFF, 0xFF);
        SDL_RenderClear(renderer);

        draw_cartpole(state(0), state(3));
        draw_goal(0, 0.5);

        //Update screen
        SDL_RenderPresent(renderer);

        SDL_Delay(dt * 1000);
#endif
    }

    /* The rhs of x' = f(x) */
    void dynamics(const std::vector<double>& x, std::vector<double>& dx, double t, const Eigen::VectorXd& u) const
    {
        double l = 0.5, m = 0.5, M = 0.5, g = 9.82, b = 0.1;

        dx[0] = x[1];
        dx[1] = (2 * m * l * std::pow(x[2], 2.0) * std::sin(x[3]) + 3 * m * g * std::sin(x[3]) * std::cos(x[3]) + 4 * u(0) - 4 * b * x[1]) / (4 * (M + m) - 3 * m * std::pow(std::cos(x[3]), 2.0));
        dx[2] = (-3 * m * l * std::pow(x[2], 2.0) * std::sin(x[3]) * std::cos(x[3]) - 6 * (M + m) * g * std::sin(x[3]) - 6 * (u(0) - b * x[1]) * std::cos(x[3])) / (4 * l * (m + M) - 3 * m * l * std::pow(std::cos(x[3]), 2.0));
        dx[3] = x[2];
    }
};

struct RewardFunction : public blackdrops::reward::Reward<RewardFunction> {
    template <typename RolloutInfo>
    double operator()(const RolloutInfo& info, const Eigen::VectorXd& from_state, const Eigen::VectorXd& action, const Eigen::VectorXd& to_state) const
    {
        double s_c_sq = 0.25 * 0.25;

        double x = to_state(0);
        double theta = to_state(3);
        double l = 0.5;

        double derr = x * x + 2. * x * l * std::sin(theta) + 2. * l * l + 2. * l * l * std::cos(theta);
        return std::exp(-0.5 / s_c_sq * derr);
    }
};

BO_DECLARE_DYN_PARAM(int, PolicyParams::nn_policy, hidden_neurons);
BO_DECLARE_DYN_PARAM(double, Params::blackdrops, boundary);
BO_DECLARE_DYN_PARAM(bool, Params::blackdrops, verbose);
BO_DECLARE_DYN_PARAM(bool, Params::blackdrops, stochastic);
BO_DECLARE_DYN_PARAM(int, Params::blackdrops, opt_evals);

BO_DECLARE_DYN_PARAM(int, Params::opt_cmaes, max_fun_evals);
BO_DECLARE_DYN_PARAM(double, Params::opt_cmaes, fun_tolerance);
BO_DECLARE_DYN_PARAM(double, Params::opt_cmaes, lbound);
BO_DECLARE_DYN_PARAM(double, Params::opt_cmaes, ubound);
BO_DECLARE_DYN_PARAM(int, Params::opt_cmaes, restarts);
BO_DECLARE_DYN_PARAM(int, Params::opt_cmaes, elitism);
BO_DECLARE_DYN_PARAM(int, Params::opt_cmaes, lambda);
BO_DECLARE_DYN_PARAM(bool, Params::opt_cmaes, handle_uncertainty);

class CartpoleArgs : public utils::CmdArgs {
public:
    CartpoleArgs() : utils::CmdArgs()
    {
        // clang-format off
        this->_desc.add_options()
                    ("opt_evals,o", po::value<int>(), "Number of rollouts for policy evaluation. Defaults to 5.");
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
                Params::blackdrops::set_opt_evals(5);
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
    CartpoleArgs cmd_arguments;
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

#if defined(USE_SDL) && !defined(NODSP)
    //Initialize
    if (!sdl_init()) {
        return 1;
    }
#endif

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
    std::cout << "  Type: Neural Network with 1 hidden layer and " << PolicyParams::nn_policy::hidden_neurons() << " hidden neurons." << std::endl;
    std::cout << std::endl;

    using policy_opt_t = limbo::opt::Cmaes<Params>;
    using kernel_t = limbo::kernel::SquaredExpARD<Params>;
    using mean_t = limbo::mean::Constant<Params>;

    using GP_t = blackdrops::model::MultiGP<Params, limbo::model::GP, kernel_t, mean_t, blackdrops::model::multi_gp::MultiGPParallelLFOpt<Params, blackdrops::model::gp::KernelLFOpt<Params, limbo::opt::Rprop<Params>>>>;

    using MGP_t = blackdrops::GPModel<Params, GP_t>;

    blackdrops::BlackDROPS<Params, MGP_t, CartPole, blackdrops::policy::NNPolicy<PolicyParams>, policy_opt_t, RewardFunction> cp_system;

    cp_system.learn(1, 15);

#if defined(USE_SDL) && !defined(NODSP)
    sdl_clean();
#endif

    return 0;
}
