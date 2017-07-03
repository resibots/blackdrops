#include <limbo/limbo.hpp>
#include <limbo/mean/constant.hpp>

#include <boost/numeric/odeint.hpp>
#include <boost/program_options.hpp>

#include <blackdrops/blackdrops.hpp>
#include <blackdrops/gp_model.hpp>
#include <blackdrops/gp_multi_model.hpp>
#include <blackdrops/model/gp/kernel_lf_opt.hpp>
#include <blackdrops/model/multi_gp.hpp>
#include <blackdrops/model/multi_gp/multi_gp_parallel_opt.hpp>
#include <blackdrops/model/multi_gp/multi_gp_whole_opt.hpp>
#include <limbo/experimental/model/poegp.hpp>
#include <limbo/experimental/model/poegp/poegp_lf_opt.hpp>

#include <blackdrops/policy/gp_policy.hpp>
#include <blackdrops/policy/nn_policy.hpp>

#include <utils/utils.hpp>

#if defined(USE_SDL) && !defined(NODSP)
#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>

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
    SDL_Rect outlineRect = {static_cast<int>(SCREEN_WIDTH / 2 - 0.05 * SCREEN_HEIGHT / 4 + x * SCREEN_HEIGHT / 4), static_cast<int>((1 - y) * SCREEN_HEIGHT / 4 - 0.05 * SCREEN_HEIGHT / 4), static_cast<int>(0.1 * SCREEN_HEIGHT / 4), static_cast<int>(0.1 * SCREEN_HEIGHT / 4)};
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
    BO_DYN_PARAM(size_t, parallel_evaluations);

    BO_PARAM(double, goal_pos, M_PI);
    BO_PARAM(double, goal_vel, 0.0);
    BO_PARAM(double, goal_pos_x, 0.0);
    BO_PARAM(double, goal_vel_x, 0.0);

    struct blackdrops {
        BO_PARAM(size_t, action_dim, 1);
        BO_PARAM(size_t, state_full_dim, 6);
        BO_PARAM(size_t, model_input_dim, 5);
        BO_PARAM(size_t, model_pred_dim, 4);
        BO_PARAM(size_t, rollout_steps, 40);
        BO_DYN_PARAM(double, boundary);
        BO_DYN_PARAM(bool, verbose);
    };

    struct options {
        BO_PARAM(bool, bounded, true);
    };

    struct gp_model {
        BO_PARAM(double, noise, 0.01);
    };

    struct model_poegp : public limbo::defaults::model_poegp {
        BO_PARAM(int, leaf_size, 100);
        BO_PARAM(double, tau, 0.05);
    };

    struct model_gpmm : public ::blackdrops::defaults::model_gpmm {
        BO_PARAM(int, threshold, 200);
    };

    struct mean_constant {
        BO_PARAM(double, constant, 0.0);
    };

    struct opt_nloptgrad : public limbo::defaults::opt_nloptgrad {
        BO_PARAM(int, iterations, 1000);
    };

    struct kernel : public limbo::defaults::kernel {
        BO_PARAM(double, noise, gp_model::noise());
        BO_PARAM(bool, optimize_noise, true);
    };

    struct kernel_squared_exp_ard : public limbo::defaults::kernel_squared_exp_ard {
    };

    struct opt_rprop : public limbo::defaults::opt_rprop {
        BO_PARAM(int, iterations, 150);
    };

    struct opt_parallelrepeater : public limbo::defaults::opt_parallelrepeater {
        BO_PARAM(int, repeats, 2);
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

    struct mean_function {
        BO_DYN_PARAM(double, pole_length);
        BO_DYN_PARAM(double, pole_mass);
        BO_DYN_PARAM(double, cart_mass);
        BO_DYN_PARAM(double, friction);
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

    struct gp_policy {
        BO_PARAM(size_t, state_dim, Params::blackdrops::model_input_dim());
        BO_PARAM(size_t, action_dim, Params::blackdrops::action_dim());
        BO_PARAM_ARRAY(double, max_u, 10.0);
        BO_PARAM(double, pseudo_samples, 10);
        BO_PARAM(double, noise, 1e-5);
        BO_PARAM_ARRAY(double, limits, 5., 5., 10., 1., 1.);
    };

    struct kernel : public limbo::defaults::kernel {
        BO_PARAM(double, noise, gp_policy::noise());
    };

    struct kernel_squared_exp_ard : public limbo::defaults::kernel_squared_exp_ard {
    };
};

struct CartPole {
    typedef std::vector<double> ode_state_type;

    template <typename Policy, typename Reward>
    std::vector<std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd>> execute(const Policy& policy, const Reward& world, size_t steps, std::vector<double>& R, bool display = true)
    {
        double dt = 0.1;
        std::vector<std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd>> res;

        boost::numeric::odeint::runge_kutta4<ode_state_type> ode_stepper;
        double t = 0.0;
        R = std::vector<double>();

        ode_state_type cp_state(4, 0.0);
        // Policy policy;
        // Eigen::VectorXd params(6);
        // params << 0.6717, 0.2685, 0.0066, 0.6987, 0.4845, 3.1517;
        // policy.set_params(params);

        for (size_t i = 0; i < steps; i++) {
            Eigen::VectorXd init(Params::blackdrops::model_input_dim());
            init(0) = cp_state[0];
            init(1) = cp_state[1];
            init(2) = cp_state[2];
            init(3) = std::cos(cp_state[3]);
            init(4) = std::sin(cp_state[3]);

            Eigen::VectorXd init_diff = Eigen::VectorXd::Map(cp_state.data(), cp_state.size());
            // while (init_diff(1) < -M_PI)
            //     init_diff(1) += 2 * M_PI;
            // while (init_diff(1) > M_PI)
            //     init_diff(1) -= 2 * M_PI;

            _u = policy.next(init)[0];
            // ode_stepper.do_step(std::bind(&CartPole::dynamics, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3), cp_state, t, dt);
            boost::numeric::odeint::integrate_const(ode_stepper,
                std::bind(&CartPole::dynamics, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3),
                cp_state, t, t + dt, dt / 2.0);
            t += dt;
            // if (cp_state[0] < -2)
            //     cp_state[0] = -2;
            // if (cp_state[0] > 2)
            //     cp_state[0] = 2;

            // TODO: Revisar como el cos y el sin estan siendo usando
            Eigen::VectorXd final = Eigen::VectorXd::Map(cp_state.data(), cp_state.size());
            // while (final(1) < -M_PI)
            //     final(1) += 2 * M_PI;
            // while (final(1) > M_PI)
            //     final(1) -= 2 * M_PI;
            Eigen::VectorXd pred = final - init_diff;
            while (final(3) < -M_PI)
                final(3) += 2 * M_PI;
            while (final(3) > M_PI)
                final(3) -= 2 * M_PI;
            res.push_back(std::make_tuple(init, limbo::tools::make_vector(_u), pred));

            // MeanIntact<Params> m;
            // Eigen::VectorXd q(6);
            // q.segment(0,5) = init.segment(0,5);
            // q(5) = _u;
            // Eigen::VectorXd kk = m.eval(q);
            // std::cout << "diff " << (final-init_diff).transpose() - kk.transpose() << std::endl;

            double r = world(init, limbo::tools::make_vector(_u), final);
            R.push_back(r);
#if defined(USE_SDL) && !defined(NODSP)
            if (display) {
                //Clear screen
                SDL_SetRenderDrawColor(renderer, 0xFF, 0xFF, 0xFF, 0xFF);
                SDL_RenderClear(renderer);

                draw_cartpole(cp_state[0], cp_state[3]);
                draw_goal(0, -0.5);

                SDL_SetRenderDrawColor(renderer, 0x00, 0xFF, 0x00, 0xFF);
                SDL_Rect outlineRect = {static_cast<int>(SCREEN_WIDTH / 2 + 0.05 * SCREEN_HEIGHT / 4), static_cast<int>(SCREEN_HEIGHT / 4 + 2.05 * SCREEN_HEIGHT / 4), static_cast<int>(_u / 10.0 * SCREEN_HEIGHT / 4), static_cast<int>(0.1 * SCREEN_HEIGHT / 4)};
                SDL_RenderFillRect(renderer, &outlineRect);

                SDL_SetRenderDrawColor(renderer, 0x00, 0xFF, 0xFF, 0xFF);
                outlineRect = {static_cast<int>(SCREEN_WIDTH / 2 + 0.05 * SCREEN_HEIGHT / 4), static_cast<int>(SCREEN_HEIGHT / 4 + 2.55 * SCREEN_HEIGHT / 4), static_cast<int>(r * SCREEN_HEIGHT / 4), static_cast<int>(0.1 * SCREEN_HEIGHT / 4)};
                SDL_RenderFillRect(renderer, &outlineRect);

                //Update screen
                SDL_RenderPresent(renderer);

                SDL_Delay(dt * 1000);
            }
#endif
        }

        if (!policy.random() && display) {
            // global::_tried_policies.push_back(policy.params());
            double rr = std::accumulate(R.begin(), R.end(), 0.0);
            std::cout << "Reward: " << rr << std::endl;
            // global::_tried_rewards.push_back(limbo::tools::make_vector(rr));
        }

        return res;
    }

    template <typename Policy, typename Model, typename Reward>
    void execute_dummy(const Policy& policy, const Model& model, const Reward& world, size_t steps, std::vector<double>& R, bool display = true) const
    {
        R = std::vector<double>();
        // init state
        Eigen::VectorXd init_diff = Eigen::VectorXd::Zero(Params::blackdrops::model_pred_dim());
        Eigen::VectorXd init = Eigen::VectorXd::Zero(Params::blackdrops::model_input_dim());
        init(3) = std::cos(0.0);
        init(4) = std::sin(0.0);
        for (size_t j = 0; j < steps; j++) {
            Eigen::VectorXd query_vec(Params::blackdrops::model_input_dim() + Params::blackdrops::action_dim());
            Eigen::VectorXd u = policy.next(init);
            query_vec.head(Params::blackdrops::model_input_dim()) = init;
            query_vec.tail(Params::blackdrops::action_dim()) = u;

            Eigen::VectorXd mu;
            Eigen::VectorXd sigma;
            std::tie(mu, sigma) = model.predictm(query_vec);
            // sigma = std::sqrt(sigma);
            // for (int i = 0; i < mu.size(); i++) {
            //     double s = gaussian_rand(mu(i), sigma);
            //     mu(i) = std::max(mu(i) - sigma, std::min(s, mu(i) + sigma));
            // }

            Eigen::VectorXd final = init_diff + mu;
            // if (final(0) < -2)
            //     final(0) = -2;
            // if (final(0) > 2)
            //     final(0) = 2;
            // while (final(1) < -M_PI)
            //     final(1) += 2 * M_PI;
            // while (final(1) > M_PI)
            //     final(1) -= 2 * M_PI;
            double r = world(init_diff, mu, final);
            R.push_back(r);
            // reward += world(init_diff, u, final);
            init_diff = final;
            init(0) = final(0);
            init(1) = final(1);
            init(2) = final(2);
            init(3) = std::cos(final(3));
            init(4) = std::sin(final(3));

#if defined(USE_SDL) && !defined(NODSP)
            if (display) {
                double dt = 0.1;
                //Clear screen
                SDL_SetRenderDrawColor(renderer, 0xFF, 0xFF, 0xFF, 0xFF);
                SDL_RenderClear(renderer);

                draw_cartpole(init_diff[0], init_diff[3], true);
                draw_goal(0, -0.5);

                SDL_SetRenderDrawColor(renderer, 0x00, 0xFF, 0x00, 0xFF);
                SDL_Rect outlineRect = {static_cast<int>(SCREEN_WIDTH / 2 + 0.05 * SCREEN_HEIGHT / 4), static_cast<int>(SCREEN_HEIGHT / 4 + 2.05 * SCREEN_HEIGHT / 4), static_cast<int>(u[0] / 10.0 * SCREEN_HEIGHT / 4), static_cast<int>(0.1 * SCREEN_HEIGHT / 4)};
                SDL_RenderFillRect(renderer, &outlineRect);

                SDL_SetRenderDrawColor(renderer, 0x00, 0xFF, 0xFF, 0xFF);
                outlineRect = {static_cast<int>(SCREEN_WIDTH / 2 + 0.05 * SCREEN_HEIGHT / 4), static_cast<int>(SCREEN_HEIGHT / 4 + 2.55 * SCREEN_HEIGHT / 4), static_cast<int>(r * SCREEN_HEIGHT / 4), static_cast<int>(0.1 * SCREEN_HEIGHT / 4)};
                SDL_RenderFillRect(renderer, &outlineRect);

                //Update screen
                SDL_RenderPresent(renderer);

                SDL_Delay(dt * 1000);
            }
#endif
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
            Eigen::VectorXd init_diff = Eigen::VectorXd::Zero(Params::blackdrops::model_pred_dim());
            Eigen::VectorXd init = Eigen::VectorXd::Zero(Params::blackdrops::model_input_dim());
            init(3) = std::cos(0.0);
            init(4) = std::sin(0.0);
            for (size_t j = 0; j < steps; j++) {
                if (init.norm() > 1000)
                    break;
                Eigen::VectorXd query_vec(Params::blackdrops::model_input_dim() + Params::blackdrops::action_dim());
                Eigen::VectorXd u = policy.next(init);
                query_vec.head(Params::blackdrops::model_input_dim()) = init;
                query_vec.tail(Params::blackdrops::action_dim()) = u;

                Eigen::VectorXd mu;
                Eigen::VectorXd sigma;
                std::tie(mu, sigma) = model.predictm(query_vec);

                if (Params::parallel_evaluations() > 1 || Params::opt_cmaes::handle_uncertainty()) {
                    sigma = sigma.array().sqrt();
                    for (int i = 0; i < mu.size(); i++) {
                        if (sigma(i) < 1e-6)
                            continue;
                        double s = gaussian_rand(mu(i), sigma(i));
                        mu(i) = std::max(mu(i) - sigma(i),
                            std::min(s, mu(i) + sigma(i)));
                    }
                }

                Eigen::VectorXd final = init_diff + mu;
                // if(final(0) < -2)
                //     final(0) = -2;
                // if(final(0) > 2)
                //     final(0) = 2;
                // while (final(1) < -M_PI)
                //     final(1) += 2 * M_PI;
                // while (final(1) > M_PI)
                //     final(1) -= 2 * M_PI;
                reward += world(init_diff, u, final);
                init_diff = final;
                init(0) = final(0);
                init(1) = final(1);
                init(2) = final(2);
                init(3) = std::cos(final(3));
                init(4) = std::sin(final(3));
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

    /* The rhs of x' = f(x) */
    void dynamics(const ode_state_type& x, ode_state_type& dx, double t)
    {
        double l = 0.5, m = 0.5, M = 0.5, g = 9.82, b = 0.1;

        dx[0] = x[1];
        dx[1] = (2 * m * l * std::pow(x[2], 2.0) * std::sin(x[3]) + 3 * m * g * std::sin(x[3]) * std::cos(x[3]) + 4 * _u - 4 * b * x[1]) / (4 * (M + m) - 3 * m * std::pow(std::cos(x[3]), 2.0));
        dx[2] = (-3 * m * l * std::pow(x[2], 2.0) * std::sin(x[3]) * std::cos(x[3]) - 6 * (M + m) * g * std::sin(x[3]) - 6 * (_u - b * x[1]) * std::cos(x[3])) / (4 * l * (m + M) - 3 * m * l * std::pow(std::cos(x[3]), 2.0));
        dx[3] = x[2];
        // dx[0] = x[1];
        // dx[1] = (_u + m * std::sin(x[3]) * (l * x[2] * x[2] + g * std::cos(x[3]))) / (M + m * std::cos(x[3]) * std::cos(x[3]));
        // dx[2] = (-_u * std::cos(x[3]) - m * l * x[2] * x[2] * std::cos(x[3]) * std::sin(x[3]) - (M + m) * g * std::sin(x[3])) / (l * (M + m * std::sin(x[3]) * std::sin(x[3])));
        // dx[3] = x[2];
    }

protected:
    double _u;
};

struct RewardFunction {
    double operator()(const Eigen::VectorXd& from_state, const Eigen::VectorXd& action, const Eigen::VectorXd& to_state) const
    {
        double s_c_sq = 0.25 * 0.25;
        // double da = std::numeric_limits<double>::max();
        // if (std::abs(to_state(3)) < 100.0)
        //     da = angle_dist(to_state(3), Params::goal_pos());
        // // double dsin = std::sin(to_state(3)); // - std::sin(Params::goal_pos());
        // // double dcos = std::cos(to_state(3)); // - std::cos(Params::goal_pos());
        // // double dy = to_state(2) - Params::goal_vel();
        // // double dz = to_state(1) - Params::goal_vel_x();
        // double dx = to_state(0); // - Params::goal_pos_x();
        // // exp(-0.5/sigma*(Δcos^2/4 + Δx*(Δsin/2 + Δx) + Δsin*(Δsin/4 + Δx/2))
        // // double derr = (dcos * dcos) / 4.0 + dsin * (dsin / 4.0 + dx / 2.0) + dx * (dsin / 2.0 + dx);
        // // double derr = dx * dx + 2 * dx * 0.5 * dsin + 2 * 0.5 * 0.5 + 2 * 0.5 * 0.5 * dcos;
        // // return std::exp(-0.5 / s_c_sq * derr);
        //
        // return std::exp(-0.5 / s_c_sq * (dx * dx + da * da));

        double x = to_state(0);
        double theta = to_state(3);
        double l = 0.5;

        double derr = x * x + 2. * x * l * std::sin(theta) + 2. * l * l + 2. * l * l * std::cos(theta);
        return std::exp(-0.5 / s_c_sq * derr);
    }
};

#ifdef MEAN
struct MeanFunc {
    typedef std::vector<double> ode_state_type;

    MeanFunc(int dim_out = 1)
    {
        _params = Eigen::VectorXd::Zero(3);
        // _params(0) = 0.45; //l
        // _params(1) = 0.55; //m
        // _params(2) = 0.4; //M
        // b = 0.2;
        _params << Params::mean_function::pole_length(), Params::mean_function::pole_mass(), Params::mean_function::cart_mass();
    }

    template <typename GP>
    Eigen::VectorXd operator()(const Eigen::VectorXd& v, const GP& gp) const
    {
        double dt = 0.1;
        boost::numeric::odeint::runge_kutta4<ode_state_type> ode_stepper;

        ode_state_type cp_state(4, 0.0);
        cp_state[0] = v(0);
        cp_state[1] = v(1);
        cp_state[2] = v(2);
        cp_state[3] = std::atan2(v(4), v(3));
        double u = v(5);

        Eigen::VectorXd init = Eigen::VectorXd::Map(cp_state.data(), cp_state.size());

        boost::numeric::odeint::integrate_const(ode_stepper,
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

    /* The rhs of x' = f(x) */
    void dynamics(const ode_state_type& x, ode_state_type& dx, double t, double u) const
    {
        double l = _params(0), m = _params(1), M = _params(2), g = 9.82, b = Params::mean_function::friction();

        dx[0] = x[1];
        dx[1] = (2 * m * l * std::pow(x[2], 2.0) * std::sin(x[3]) + 3 * m * g * std::sin(x[3]) * std::cos(x[3]) + 4 * u - 4 * b * x[1]) / (4 * (M + m) - 3 * m * std::pow(std::cos(x[3]), 2.0));
        dx[2] = (-3 * m * l * std::pow(x[2], 2.0) * std::sin(x[3]) * std::cos(x[3]) - 6 * (M + m) * g * std::sin(x[3]) - 6 * (u - b * x[1]) * std::cos(x[3])) / (4 * l * (m + M) - 3 * m * l * std::pow(std::cos(x[3]), 2.0));
        dx[3] = x[2];
    }

protected:
    Eigen::VectorXd _params;
};
#endif

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

BO_DECLARE_DYN_PARAM(double, Params::mean_function, pole_length);
BO_DECLARE_DYN_PARAM(double, Params::mean_function, pole_mass);
BO_DECLARE_DYN_PARAM(double, Params::mean_function, cart_mass);
BO_DECLARE_DYN_PARAM(double, Params::mean_function, friction);

int main(int argc, char** argv)
{
    bool uncertainty = false;
    bool verbose = false;
    int threads = tbb::task_scheduler_init::automatic;
    namespace po = boost::program_options;
    po::options_description desc("Command line arguments");
    // clang-format off
    desc.add_options()("help,h", "Prints this help message")
                      ("parallel_evaluations,p", po::value<int>(), "Number of parallel monte carlo evaluations for policy reward estimation.")
                      ("hidden_neurons,n", po::value<int>(), "Number of hidden neurons in NN policy.")
                      ("boundary,b", po::value<double>(), "Boundary of the values during the optimization.")
                      ("max_evals,m", po::value<int>(), "Max function evaluations to optimize the policy.")
                      ("tolerance,t", po::value<double>(), "Maximum tolerance to continue optimizing the function.")
                      ("restarts,r", po::value<int>(), "Max number of restarts to use during optimization.")
                      ("elitism,e", po::value<int>(), "Elitism mode to use [0 to 3].")
                      ("uncertainty,u", po::bool_switch(&uncertainty)->default_value(false), "Enable uncertainty handling.")
                      ("threads,d", po::value<int>(), "Max number of threads used by TBB")
                      ("verbose,v", po::bool_switch(&verbose)->default_value(false), "Enable verbose mode.")
                      ("pole_length", po::value<double>(), "Initial length of the pole for the mean function [0 to 1].")
                      ("pole_mass", po::value<double>(), "Initial mass of the pole for the mean function [0 to 1].")
                      ("cart_mass", po::value<double>(), "Initial mass of the cart for the mean function [0 to 1].")
                      ("friction", po::value<double>(), "Initial friction coefficient for the mean function [0 to 1].");
    // clang-format on

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
        // Mean Function parameters
        if (vm.count("pole_length")) {
            double pl = vm["pole_length"].as<double>();
            if (pl < 0.0)
                pl = 0.0;
            if (pl > 1.0)
                pl = 1.0;
            Params::mean_function::set_pole_length(pl);
        }
        else {
            Params::mean_function::set_pole_length(0.5);
        }
        if (vm.count("pole_mass")) {
            double pm = vm["pole_mass"].as<double>();
            if (pm < 0.0)
                pm = 0.0;
            if (pm > 1.0)
                pm = 1.0;
            Params::mean_function::set_pole_mass(pm);
        }
        else {
            Params::mean_function::set_pole_mass(0.5);
        }
        if (vm.count("cart_mass")) {
            double cm = vm["cart_mass"].as<double>();
            if (cm < 0.0)
                cm = 0.0;
            if (cm > 1.0)
                cm = 1.0;
            Params::mean_function::set_cart_mass(cm);
        }
        else {
            Params::mean_function::set_cart_mass(0.5);
        }
        if (vm.count("friction")) {
            double fr = vm["friction"].as<double>();
            if (fr < 0.0)
                fr = 0.0;
            if (fr > 1.0)
                fr = 1.0;
            Params::mean_function::set_friction(fr);
        }
        else {
            Params::mean_function::set_friction(0.1);
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
#ifdef MEAN
    std::cout << "Mean parameters: " << std::endl;
    std::cout << "  Pole length (m): " << Params::mean_function::pole_length() << std::endl;
    std::cout << "  Pole mass (kg): " << Params::mean_function::pole_mass() << std::endl;
    std::cout << "  Cart mass (kg): " << Params::mean_function::cart_mass() << std::endl;
    std::cout << "  Friction (N/m/s): " << Params::mean_function::friction() << std::endl;
#endif

    using policy_opt_t = limbo::opt::Cmaes<Params>;

    using kernel_t = limbo::kernel::SquaredExpARD<Params>;
#ifndef MEAN
    using mean_t = limbo::mean::Constant<Params>;
#else
    using mean_t = MeanFunc;
#endif

#ifndef MODELIDENT
    using GP_t = blackdrops::model::MultiGP<Params, limbo::model::GP, kernel_t, mean_t, blackdrops::model::multi_gp::MultiGPParallelLFOpt<Params, blackdrops::model::gp::KernelLFOpt<Params>>>; //, limbo::opt::NLOptGrad<Params, nlopt::LD_SLSQP>>>;
#else
    using GP_t = blackdrops::model::MultiGP<Params, limbo::model::GP, kernel_t, mean_t, blackdrops::model::multi_gp::MultiGPWholeLFOpt<Params, limbo::opt::NLOptNoGrad<Params, nlopt::LN_SBPLX>>>;
#endif

#ifdef SPGPS
#ifndef MODELIDENT
    using SPGP_t = blackdrops::model::MultiGP<Params, limbo::experimental::model::POEGP, kernel_t, mean_t, blackdrops::model::multi_gp::MultiGPParallelLFOpt<Params, limbo::experimental::model::poegp::POEKernelLFOpt<Params>>>;
#else
    using SPGP_t = blackdrops::model::MultiGP<Params, limbo::experimental::model::POEGP, kernel_t, mean_t, blackdrops::model::multi_gp::MultiGPWholeLFOpt<Params, limbo::opt::NLOptNoGrad<Params, nlopt::LN_SBPLX>, limbo::experimental::model::poegp::POEKernelLFOpt<Params>>>;
#endif
    using GPMM_t = blackdrops::GPMultiModel<Params, GP_t, SPGP_t>;
    using MGP_t = blackdrops::GPModel<Params, GPMM_t>;
#else
    using MGP_t = blackdrops::GPModel<Params, GP_t>;
#endif

#ifndef GPPOLICY
    blackdrops::BlackDROPS<Params, MGP_t, CartPole, blackdrops::policy::NNPolicy<PolicyParams>, policy_opt_t, RewardFunction> cp_system;
#else
    blackdrops::BlackDROPS<Params, MGP_t, CartPole, blackdrops::policy::GPPolicy<PolicyParams>, policy_opt_t, RewardFunction> cp_system;
#endif

    cp_system.learn(1, 15);

#if defined(USE_SDL) && !defined(NODSP)
    sdl_clean();
#endif

    return 0;
}
