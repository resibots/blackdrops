#include <limbo/limbo.hpp>

#include <boost/numeric/odeint.hpp>
#include <boost/program_options.hpp>

#include <blackdrops/blackdrops.hpp>

#include <blackdrops/gp_model.hpp>
#include <blackdrops/model/gp/kernel_lf_opt.hpp>
#include <blackdrops/model/multi_gp.hpp>
#include <blackdrops/model/multi_gp/multi_gp_parallel_opt.hpp>
#include <blackdrops/model/multi_gp/multi_gp_whole_opt.hpp>
#include <blackdrops/system/ode_system.hpp>

#include <blackdrops/mi_model.hpp>

#include <blackdrops/policy/nn_policy.hpp>

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

    window = SDL_CreateWindow("Pendubot Task", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, SCREEN_WIDTH, SCREEN_HEIGHT, SDL_WINDOW_SHOWN);
    if (window == NULL) {
        std::cout << "Window could not be created! SDL_Error: " << SDL_GetError() << std::endl;
        return false;
    }

    //Create renderer for window
    renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
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

bool draw_pendulum(double theta1, double theta2, bool red = false)
{
    double l = 0.5;
    // double c1 = std::cos(theta1), s1 = std::sin(theta1);
    // double c12 = std::cos(theta1 + theta2), s12 = std::sin(theta1 + theta2);

    double x1 = -l * std::sin(theta1), y1 = l * std::cos(theta1);
    double x2 = -l * (std::sin(theta1) + std::sin(theta2)), y2 = l * (std::cos(theta1) + std::cos(theta2));

    //Draw blue horizontal line
    SDL_SetRenderDrawColor(renderer, 0x00, 0x00, 0xFF, 0xFF);
    if (red)
        SDL_SetRenderDrawColor(renderer, 0xFF, 0x00, 0x00, 0xFF);
    SDL_Rect outlineRect = {static_cast<int>(SCREEN_WIDTH / 2 - 0.05 * SCREEN_HEIGHT / 4), static_cast<int>(SCREEN_HEIGHT / 2 - 0.05 * SCREEN_HEIGHT / 4), static_cast<int>(0.1 * SCREEN_HEIGHT / 4), static_cast<int>(0.1 * SCREEN_HEIGHT / 4)};
    SDL_RenderFillRect(renderer, &outlineRect);
    outlineRect = {SCREEN_WIDTH / 2 + x1 * SCREEN_HEIGHT / 4 - 0.05 * SCREEN_HEIGHT / 4, SCREEN_HEIGHT / 2 - y1 * SCREEN_HEIGHT / 4 - 0.05 * SCREEN_HEIGHT / 4, static_cast<int>(0.1 * SCREEN_HEIGHT / 4), static_cast<int>(0.1 * SCREEN_HEIGHT / 4)};
    SDL_RenderFillRect(renderer, &outlineRect);
    SDL_RenderDrawLine(renderer, SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2, static_cast<int>(SCREEN_WIDTH / 2 + x1 * SCREEN_HEIGHT / 4), static_cast<int>(SCREEN_HEIGHT / 2 - y1 * SCREEN_HEIGHT / 4));
    SDL_RenderDrawLine(renderer, static_cast<int>(SCREEN_WIDTH / 2 + x1 * SCREEN_HEIGHT / 4), static_cast<int>(SCREEN_HEIGHT / 2 - y1 * SCREEN_HEIGHT / 4), static_cast<int>(SCREEN_WIDTH / 2 + x2 * SCREEN_HEIGHT / 4), static_cast<int>(SCREEN_HEIGHT / 2 - y2 * SCREEN_HEIGHT / 4));

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
    if (renderer)
        SDL_DestroyRenderer(renderer);
    //Destroy window
    if (window)
        SDL_DestroyWindow(window);
    //Quit SDL
    SDL_Quit();
}
#endif

struct Params {
    struct blackdrops {
        BO_PARAM(size_t, action_dim, 1);
        BO_PARAM(size_t, model_input_dim, 6);
        BO_PARAM(size_t, model_pred_dim, 4);
        BO_PARAM(double, dt, 0.05);
        BO_PARAM(double, T, 2.5);
        BO_DYN_PARAM(bool, verbose);
        BO_DYN_PARAM(double, boundary);
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
        BO_PARAM(int, verbose, false);
        BO_PARAM(bool, fun_compute_initial, true);
        // BO_PARAM(double, fun_target, 30);
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

    struct opt_nloptnograd : public limbo::defaults::opt_nloptnograd {
#ifndef ONLYMI
        BO_PARAM(int, iterations, 1000);
        BO_PARAM(double, fun_tolerance, 1e-4);
        BO_PARAM(double, xrel_tolerance, 1e-4);
#else
        BO_PARAM(int, iterations, 2000);
        BO_PARAM(double, fun_tolerance, 1e-6);
        BO_PARAM(double, xrel_tolerance, 1e-6);
#endif
    };

    struct mean_function {
        BO_DYN_PARAM(double, mass1);
        BO_DYN_PARAM(double, mass2);
        BO_DYN_PARAM(double, length1);
        BO_DYN_PARAM(double, length2);
        BO_DYN_PARAM(double, friction1);
        BO_DYN_PARAM(double, friction2);
    };
};

struct PolicyParams {
    struct blackdrops : public Params::blackdrops {
    };

    struct nn_policy {
        BO_PARAM(size_t, state_dim, Params::blackdrops::model_input_dim());
        BO_PARAM(size_t, action_dim, Params::blackdrops::action_dim());
        BO_PARAM_ARRAY(double, max_u, 3.5);
        BO_DYN_PARAM(int, hidden_neurons);
        BO_PARAM_ARRAY(double, limits, 10., 10., 1., 1., 1., 1.);
        BO_PARAM(double, af, 5.0);
    };
};

Eigen::VectorXd tip(double theta1, double theta2)
{
    double l = 0.5;

    double y = l * (std::cos(theta1) + std::cos(theta2));
    double x = -l * (std::sin(theta1) + std::sin(theta2));

    Eigen::VectorXd ret(2);
    ret << x, y;

    return ret;
}

struct Pendubot : public blackdrops::system::ODESystem<Params> {
    typedef std::vector<double> ode_state_type;

    Eigen::VectorXd init_state() const
    {
        Eigen::VectorXd init = Eigen::VectorXd::Zero(4);
        init(2) = M_PI;
        init(3) = M_PI;

        return init;
    }

    Eigen::VectorXd transform_state(const Eigen::VectorXd& original_state) const
    {
        Eigen::VectorXd trans_state = Eigen::VectorXd::Zero(6);
        trans_state.head(2) = original_state.head(2);
        trans_state(2) = std::cos(original_state(2));
        trans_state(3) = std::sin(original_state(2));
        trans_state(4) = std::cos(original_state(3));
        trans_state(5) = std::sin(original_state(3));

        return trans_state;
    }

    void draw_single(const Eigen::VectorXd& state) const
    {
#if defined(USE_SDL) && !defined(NODSP)
        double dt = Params::blackdrops::dt();
        //Clear screen
        SDL_SetRenderDrawColor(renderer, 0xFF, 0xFF, 0xFF, 0xFF);
        SDL_RenderClear(renderer);

        draw_pendulum(state(2), state(3));
        draw_goal(0, 1);

        //Update screen
        SDL_RenderPresent(renderer);

        SDL_Delay(dt * 1000);
#endif
    }

    /* The rhs of x' = f(x) */
    void dynamics(const ode_state_type& x, ode_state_type& dx, double t, const Eigen::VectorXd& u) const
    {
        double m1 = 0.5; // [kg]     mass of 1st link
        double m2 = 0.5; // [kg]     mass of 2nd link
        double b1 = 0.1; // [Ns/m]   coefficient of friction (1st joint)
        double b2 = 0.1; // [Ns/m]   coefficient of friction (2nd joint)
        double l1 = 0.5; // [m]      length of 1st pendulum
        double l2 = 0.5; // [m]      length of 2nd pendulum
        double g = 9.82; // [m/s^2]  acceleration of gravity
        double I1 = m1 * l1 * l1 / 12.; // moment of inertia around pendulum midpoint (1st link)
        double I2 = m2 * l2 * l2 / 12.; // moment of inertia around pendulum midpoint (2nd link)
        double u1 = u(0);

        Eigen::MatrixXd A(2, 2);
        A << l1 * l1 * (0.25 * m1 + m2) + I1, 0.5 * m2 * l1 * l2 * std::cos(x[2] - x[3]),
            0.5 * m2 * l1 * l2 * std::cos(x[2] - x[3]), l2 * l2 * 0.25 * m2 + I2;
        Eigen::VectorXd b(2);
        b << g * l1 * std::sin(x[2]) * (0.5 * m1 + m2) - 0.5 * m2 * l1 * l2 * x[1] * x[1] * std::sin(x[2] - x[3]) + u1 - b1 * x[0],
            0.5 * m2 * l2 * (l1 * x[0] * x[0] * std::sin(x[2] - x[3]) + g * std::sin(x[3])) - b2 * x[1];
        Eigen::VectorXd sol = A.llt().solve(b);

        dx[0] = sol(0);
        dx[1] = sol(1);
        dx[2] = x[0];
        dx[3] = x[1];
    }
};

struct RewardFunction {
    double operator()(const Eigen::VectorXd& from_state, const Eigen::VectorXd& action, const Eigen::VectorXd& to_state) const
    {
        double s_c_sq = 0.5 * 0.5;

        Eigen::VectorXd tip_pos = tip(to_state(2), to_state(3));
        Eigen::VectorXd tip_goal(2);
        tip_goal << 0.0, 1.0;
        double derr = (tip_goal - tip_pos).squaredNorm();

        return std::exp(-0.5 / s_c_sq * derr);

        //$1-\exp(-0.5*d^2*a)$,  where $a>0$ and  $d^2$ is the squared difference
        // between the actual and desired position of the tip of the outer pendulum.
    }
};

#ifdef MEAN
struct MeanFunc {
    MeanFunc(int dim_out = 1)
    {
        _params = Eigen::VectorXd::Zero(4);
        _params << Params::mean_function::mass1(), Params::mean_function::mass2(), Params::mean_function::length1(), Params::mean_function::length2();
    }

    template <typename GP>
    Eigen::VectorXd operator()(const Eigen::VectorXd& v, const GP& gp) const
    {
        double dt = Params::blackdrops::dt();

        std::vector<double> cp_state(4, 0.0);
        cp_state[0] = v(0);
        cp_state[1] = v(1);
        cp_state[2] = std::atan2(v(3), v(2));
        cp_state[3] = std::atan2(v(5), v(4));
        double u = v(6);

        Eigen::VectorXd init = Eigen::VectorXd::Map(cp_state.data(), cp_state.size());

        boost::numeric::odeint::integrate_const(boost::numeric::odeint::make_dense_output(1e-12, 1e-12, boost::numeric::odeint::runge_kutta_dopri5<std::vector<double>>()),
            std::bind(&MeanFunc::dynamics, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, u),
            cp_state, 0.0, dt, dt / 2.0);

        Eigen::VectorXd final = Eigen::VectorXd::Map(cp_state.data(), cp_state.size());
        return (final - init);
    }

    Eigen::VectorXd h_params() const { return _params; }

    void set_h_params(const Eigen::VectorXd& params)
    {
        _params = params;
    }

    void dynamics(const std::vector<double>& x, std::vector<double>& dx, double t, double u1) const
    {
        double m1 = _params(0); // [kg]     mass of 1st link
        double m2 = _params(1); // [kg]     mass of 2nd link
        double b1 = Params::mean_function::friction1(); // [Ns/m]   coefficient of friction (1st joint)
        double b2 = Params::mean_function::friction2(); // [Ns/m]   coefficient of friction (2nd joint)
        double l1 = _params(2); // [m]      length of 1st pendulum
        double l2 = _params(3); // [m]      length of 2nd pendulum
        double g = 9.82; // [m/s^2]  acceleration of gravity
        double I1 = m1 * l1 * l1 / 12.; // moment of inertia around pendulum midpoint (1st link)
        double I2 = m2 * l2 * l2 / 12.; // moment of inertia around pendulum midpoint (2nd link)

        Eigen::MatrixXd A(2, 2);
        A << l1 * l1 * (0.25 * m1 + m2) + I1, 0.5 * m2 * l1 * l2 * std::cos(x[2] - x[3]),
            0.5 * m2 * l1 * l2 * std::cos(x[2] - x[3]), l2 * l2 * 0.25 * m2 + I2;
        Eigen::VectorXd b(2);
        b << g * l1 * std::sin(x[2]) * (0.5 * m1 + m2) - 0.5 * m2 * l1 * l2 * x[1] * x[1] * std::sin(x[2] - x[3]) + u1 - b1 * x[0],
            0.5 * m2 * l2 * (l1 * x[0] * x[0] * std::sin(x[2] - x[3]) + g * std::sin(x[3])) - b2 * x[1];
        Eigen::VectorXd sol = A.llt().solve(b);

        dx[0] = sol(0);
        dx[1] = sol(1);
        dx[2] = x[0];
        dx[3] = x[1];
    }

protected:
    Eigen::VectorXd _params;
};
#endif

BO_DECLARE_DYN_PARAM(int, PolicyParams::nn_policy, hidden_neurons);
BO_DECLARE_DYN_PARAM(bool, Params::blackdrops, verbose);
BO_DECLARE_DYN_PARAM(double, Params::blackdrops, boundary);

BO_DECLARE_DYN_PARAM(double, Params::opt_cmaes, max_fun_evals);
BO_DECLARE_DYN_PARAM(double, Params::opt_cmaes, fun_tolerance);
BO_DECLARE_DYN_PARAM(double, Params::opt_cmaes, lbound);
BO_DECLARE_DYN_PARAM(double, Params::opt_cmaes, ubound);
BO_DECLARE_DYN_PARAM(int, Params::opt_cmaes, restarts);
BO_DECLARE_DYN_PARAM(int, Params::opt_cmaes, elitism);
BO_DECLARE_DYN_PARAM(bool, Params::opt_cmaes, handle_uncertainty);

BO_DECLARE_DYN_PARAM(double, Params::mean_function, mass1);
BO_DECLARE_DYN_PARAM(double, Params::mean_function, mass2);
BO_DECLARE_DYN_PARAM(double, Params::mean_function, length1);
BO_DECLARE_DYN_PARAM(double, Params::mean_function, length2);
BO_DECLARE_DYN_PARAM(double, Params::mean_function, friction1);
BO_DECLARE_DYN_PARAM(double, Params::mean_function, friction2);

int main(int argc, char** argv)
{
    bool uncertainty = false;
    int threads = tbb::task_scheduler_init::automatic;
    bool verbose = false;
    std::string policy_file = "";

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
                      ("verbose,v", po::bool_switch(&verbose)->default_value(false), "Enable verbose mode.")
                      ("mass1", po::value<double>(), "Initial mass of the first link for the mean function [0 to 1].")
                      ("mass2", po::value<double>(), "Initial mass of the second link for the mean function [0 to 1].")
                      ("length1", po::value<double>(), "Initial length of the first link for the mean function [0 to 1].")
                      ("length2", po::value<double>(), "Initial length of the second link for the mean function [0 to 1].")
                      ("friction1", po::value<double>(), "Initial friction coefficient of the first joint for the mean function [0 to 1].")
                      ("friction2", po::value<double>(), "Initial friction coefficient of the second joint for the mean function [0 to 1].")
                      ("policy", po::value<std::string>(), "Path to load policy file");
    // clang-format on

    try {
        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help")) {
            std::cout << desc << std::endl;
            return 0;
        }

        po::notify(vm);

        if (vm.count("threads"))
            threads = vm["threads"].as<int>();
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
        if (vm.count("policy")) {
            policy_file = vm["policy"].as<std::string>();
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
        // Mean Function parameters
        if (vm.count("mass1")) {
            double m1 = vm["mass1"].as<double>();
            if (m1 < 0.0)
                m1 = 0.0;
            if (m1 > 1.0)
                m1 = 1.0;
            Params::mean_function::set_mass1(m1);
        }
        else {
            Params::mean_function::set_mass1(0.5);
        }
        if (vm.count("mass2")) {
            double m2 = vm["mass2"].as<double>();
            if (m2 < 0.0)
                m2 = 0.0;
            if (m2 > 1.0)
                m2 = 1.0;
            Params::mean_function::set_mass2(m2);
        }
        else {
            Params::mean_function::set_mass2(0.5);
        }
        if (vm.count("length1")) {
            double l1 = vm["length1"].as<double>();
            if (l1 < 0.0)
                l1 = 0.0;
            if (l1 > 1.0)
                l1 = 1.0;
            Params::mean_function::set_length1(l1);
        }
        else {
            Params::mean_function::set_length1(0.5);
        }
        if (vm.count("length2")) {
            double l2 = vm["length2"].as<double>();
            if (l2 < 0.0)
                l2 = 0.0;
            if (l2 > 1.0)
                l2 = 1.0;
            Params::mean_function::set_length2(l2);
        }
        else {
            Params::mean_function::set_length2(0.5);
        }
        if (vm.count("friction1")) {
            double fr = vm["friction1"].as<double>();
            if (fr < 0.0)
                fr = 0.0;
            if (fr > 1.0)
                fr = 1.0;
            Params::mean_function::set_friction1(fr);
        }
        else {
            Params::mean_function::set_friction1(0.1);
        }
        if (vm.count("friction2")) {
            double fr = vm["friction2"].as<double>();
            if (fr < 0.0)
                fr = 0.0;
            if (fr > 1.0)
                fr = 1.0;
            Params::mean_function::set_friction2(fr);
        }
        else {
            Params::mean_function::set_friction2(0.1);
        }
    }
    catch (po::error& e) {
        std::cerr << "[Exception caught while parsing command line arguments]: " << e.what() << std::endl;
        return 1;
    }
#if defined(USE_SDL) && !defined(NODSP)
    //Initialize
    if (!sdl_init()) {
        return 1;
    }
#endif

#ifdef USE_TBB
    static tbb::task_scheduler_init init(threads);
#endif

    Params::blackdrops::set_verbose(verbose);
    Params::opt_cmaes::set_handle_uncertainty(uncertainty);

    std::cout << "Cmaes parameters:" << std::endl;
    std::cout << "  max_fun_evals = " << Params::opt_cmaes::max_fun_evals() << std::endl;
    std::cout << "  fun_tolerance = " << Params::opt_cmaes::fun_tolerance() << std::endl;
    std::cout << "  restarts = " << Params::opt_cmaes::restarts() << std::endl;
    std::cout << "  elitism = " << Params::opt_cmaes::elitism() << std::endl;
    std::cout << "  handle_uncertainty = " << Params::opt_cmaes::handle_uncertainty() << std::endl;
    std::cout << "  boundary = " << Params::blackdrops::boundary() << std::endl;
    std::cout << std::endl;
#ifdef MEAN
    std::cout << "Mean parameters: " << std::endl;
    std::cout << "  Mass of first link (kg): " << Params::mean_function::mass1() << std::endl;
    std::cout << "  Mass of second link (kg): " << Params::mean_function::mass2() << std::endl;
    std::cout << "  Length of first link (m): " << Params::mean_function::length1() << std::endl;
    std::cout << "  Length of second link (m): " << Params::mean_function::length2() << std::endl;
    std::cout << "  Friction of first joint (N/m/s): " << Params::mean_function::friction1() << std::endl;
    std::cout << "  Friction of second joint (N/m/s): " << Params::mean_function::friction2() << std::endl;
#endif

    using kernel_t = limbo::kernel::SquaredExpARD<Params>;
#ifndef MEAN
    using mean_t = limbo::mean::Constant<Params>;
#else
    using mean_t = MeanFunc;
#endif

#ifndef MODELIDENT
    using GP_t = blackdrops::model::MultiGP<Params, limbo::model::GP, kernel_t, mean_t, blackdrops::model::multi_gp::MultiGPParallelLFOpt<Params, blackdrops::model::gp::KernelLFOpt<Params>>>;
#else
    using GP_t = blackdrops::model::MultiGP<Params, limbo::model::GP, kernel_t, mean_t, blackdrops::model::multi_gp::MultiGPWholeLFOpt<Params, limbo::opt::NLOptNoGrad<Params, nlopt::LN_SBPLX>>>;
#endif

    using policy_opt_t = limbo::opt::Cmaes<Params>;

#ifndef ONLYMI
    using MGP_t = blackdrops::GPModel<Params, GP_t>;
#else
    using MGP_t = blackdrops::MIModel<Params, mean_t, limbo::opt::NLOptNoGrad<Params, nlopt::LN_SBPLX>>;
#endif

    blackdrops::BlackDROPS<Params, MGP_t, Pendubot, blackdrops::policy::NNPolicy<PolicyParams>, policy_opt_t, RewardFunction> pend_system;

    pend_system.learn(1, 25, true, policy_file);

#if defined(USE_SDL) && !defined(NODSP)
    sdl_clean();
#endif

    return 0;
}