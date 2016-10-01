#include <iostream>
#include <string>

#include <boost/numeric/odeint.hpp>

/* The type of container used to hold the state vector */
typedef std::vector<double> state_type;

state_type pendulum_state(2);

#ifdef USE_SDL
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

    window = SDL_CreateWindow("SDL Tutorial", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, SCREEN_WIDTH, SCREEN_HEIGHT, SDL_WINDOW_SHOWN);
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

bool draw_pendulum(double theta)
{
    double x = std::cos(theta), y = std::sin(theta);

    SDL_Rect outlineRect = {SCREEN_WIDTH / 2 - 0.05 * SCREEN_HEIGHT / 4, SCREEN_HEIGHT / 2 - 0.05 * SCREEN_HEIGHT / 4, 0.1 * SCREEN_HEIGHT / 4, 0.1 * SCREEN_HEIGHT / 4};
    SDL_SetRenderDrawColor(renderer, 0x00, 0x00, 0xFF, 0xFF);
    SDL_RenderFillRect(renderer, &outlineRect);
    //Draw blue horizontal line
    SDL_SetRenderDrawColor(renderer, 0x00, 0x00, 0xFF, 0xFF);
    SDL_RenderDrawLine(renderer, SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2, SCREEN_WIDTH / 2 + y * SCREEN_HEIGHT / 4, SCREEN_HEIGHT / 2 + x * SCREEN_HEIGHT / 4);

    return true;
}

bool draw_goal(double x, double y)
{
    SDL_Rect outlineRect = {SCREEN_WIDTH / 2 - 0.05 * SCREEN_HEIGHT / 4, SCREEN_HEIGHT / 4 - 0.05 * SCREEN_HEIGHT / 4, 0.1 * SCREEN_HEIGHT / 4, 0.1 * SCREEN_HEIGHT / 4};
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

double u(double t)
{
    return 2.5;
}

/* The rhs of x' = f(x) */
void pendulum(const state_type& x, state_type& dx, double t)
{
    double l = 1, m = 1, g = 9.82, b = 0.01;

    dx[0] = (u(t) - b * x[0] - m * g * l * std::sin(x[1]) / 2.0) / (m * std::pow(l, 2) / 3.0);
    dx[1] = x[0];
}

void reset_pendulum()
{
    pendulum_state[0] = 0.0;
    pendulum_state[1] = 0.0;
}

int main(int argc, char** argv)
{
    const double dt = 0.1;
    boost::numeric::odeint::runge_kutta_dopri5<state_type> stepper;

    reset_pendulum();

    double t = 0.0;
// std::cout << x[0] << " " << x[1] << std::endl;

#ifdef USE_SDL
    //Make sure the program waits for a quit
    bool quit = false;

    //Initialize
    if (!sdl_init()) {
        return 1;
    }

    //Event handler
    SDL_Event e;

    //While the user hasn't quit
    while (!quit) {
        //Handle events on queue
        while (SDL_PollEvent(&e) != 0) {
            //User requests quit
            if (e.type == SDL_QUIT) {
                quit = true;
            }
        }

#endif

        // for (size_t i(0); i <= 100; ++i) {
        stepper.do_step(pendulum, pendulum_state, t, dt);
        t += dt;

// if (std::abs(t - 5.0) <= 1e-1) {
//     reset_pendulum();
//     t = 0.0;
//     stepper.reset();
// }
// std::cout << x[0] << " " << x[1] << std::endl;
// }

#ifdef USE_SDL
        //Clear screen
        SDL_SetRenderDrawColor(renderer, 0xFF, 0xFF, 0xFF, 0xFF);
        SDL_RenderClear(renderer);

        draw_pendulum(pendulum_state[1]);
        draw_goal(0, -1);

        //Update screen
        SDL_RenderPresent(renderer);

        SDL_Delay(dt * 1000);
    }
    sdl_clean();
#endif

    return 0;
}
