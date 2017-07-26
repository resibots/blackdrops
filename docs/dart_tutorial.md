## How to create your own DART-based scenario using Black-DROPS

In this tutorial we will go through the basic steps of how to create your own [DART](http://dartsim.github.io/)-based scenario using Black-DROPS. This tutorial builds on the [basic tutorial](basic_tutorial.md) and we assume that you have already gone though that one. As such, we will not provide step by step instructions, but only what is specifically needed for DART-based scenarios. Further more, this scenario was quickly developed and should not be taken as the solution of Black-DROPS to the problem.

### Scenario

<center>
<img src="https://i.ytimg.com/vi/Zaj9QolJMLg/hqdefault.jpg" width="400">
</center>

We will create a variant of the [reacher scenario of OpenAI Gym](https://gym.openai.com/envs/Reacher-v1); more specifically we will take the [skeleton file](http://dartsim.github.io/skel_file_format.html) of the task from [DartEnv](https://github.com/DartEnv/dart-env/blob/master/gym/envs/dart/assets/reacher2d.skel). In more detail:

- The state space is 4-D (the angular positions and velocities of the joints)
- The action space is 2-D (the torques applied to the joints)
- The reward function is unknown to Black-DROPS and has to learn it from data
- The reward function is simply the negative distance of the end-effector to the desired location
- The policy we optimize for is a simple feed-forward neural network with one hidden layer (and the hyperbolic tangent function as the activation function)
- We optimize the policy for just reaching one target and not for any target

### Tutorial

You can find the code of this scenario (already coded) in the `src/tutorials/dart_reacher2d.cpp` file.

#### Parameters

Similarly to the [basic tutorial](basic_tutorial.md), we set the following parameters:

- Number of dimensions of the state space: 4-D
- Number of dimensions of the transformed state space (to the GPs and the policy): 6-D (2 angular velocities + 2 sines + 2 cosines of angular positions)
- Number of dimensions of the action space: 2-D
- Sampling/control rate: we take the sampling/control rate of the original scenario (i.e., 100Hz)
- Duration of each episode/trial: 0.5 secs (as in the original OpenAI Gym scenario)
- Bounds of the action space: this should be [-50., 50.]
- Normalization factors for the neural network policy: [30., 70., 1., 1., 1., 1.]

#### Loading URDF/SKEL files

In order to simulate robots with DART, we rely on the [robot\_dart](https://github.com/resibots/robot_dart) wrapper. This wrapper allows us to easily load URDF files and manage simulated worlds where we want to mainly control one robot. In Black-DROPS, we also provide an easy way of loading robot from SKEL files. Here's an example for loading the skeleton (robot) with name "arm" from a skel file:

```cpp
    auto myrobot = std::make_shared<robot_dart::Robot>(utils::load_skel(robot_file, "arm"));
```

Here's an example for loading a robot from a URDF file (here "arm" will be the name of the skeleton in DART):

```cpp
    auto myrobot = std::make_shared<robot_dart::Robot>(robot_dart::Robot(robot_file, {}, "arm", true));
```

In this example, we load the robot from a skel file (we also load the floor and position things properly):

```cpp
void init_simu(const std::string& robot_file)
{
    global::global_robot = std::make_shared<robot_dart::Robot>(utils::load_skel(robot_file, "arm"));
    Eigen::Isometry3d tf = global::global_robot->skeleton()->getRootBodyNode()->getParentJoint()->getTransformFromParentBodyNode();
    tf.translation() = Eigen::Vector3d(0., 0.01, 0.);
    global::global_robot->skeleton()->getRootBodyNode()->getParentJoint()->setTransformFromParentBodyNode(tf);

    global::global_floor = utils::load_skel(robot_file, "ground skeleton");

    global::goal = Eigen::Vector3d(0.1, 0.01, -0.1); // this is the goal position (the robot moves only in x,z axes)

    std::cout << "Goal is: " << global::goal.transpose() << std::endl;
}
```

We store the goal position, the robot and the floor skeleton in the global namespace so that we can easily access them. Lastly, a variable `RESPATH` is defined to point to the `/path/to/repo/res` folder; it can be elegantly retrieved in C++ by `std::string(RESPATH)`. 

#### System struct

Next we should define our robot/system that will be simulated using DART. In Black-DROPS, this is done by defining 2 structs/classes with the following signature:

```cpp
struct PolicyControl : public blackdrops::system::BaseDARTPolicyControl<Params, global::policy_t> {
    using base_t = blackdrops::system::BaseDARTPolicyControl<Params, global::policy_t>;

    PolicyControl() : base_t() {}
    PolicyControl(const std::vector<double>& ctrl, base_t::robot_t robot) : base_t(ctrl, robot) {}

    Eigen::VectorXd get_state(const base_t::robot_t& robot, bool full) const
    {
        // here goes the code for getting the robot state
    }
};

struct MyDARTSystem : public blackdrops::system::DARTSystem<Params, PolicyControl> {
    using base_t = blackdrops::system::DARTSystem<Params, PolicyControl>;

    Eigen::VectorXd init_state() const
    {
        // same as in the ODESystem
    }

    Eigen::VectorXd transform_state(const Eigen::VectorXd& original_state) const
    {
        // same as in the ODESystem
    }

    std::shared_ptr<robot_dart::Robot> get_robot() const
    {
        // write code to get a robot for simulation
    }

    void add_extra_to_simu(base_t::robot_simu_t& simu) const
    {
        // optionally we can add extra things to the simulation
        // e.g. markers for the goal position (etc.)
    }
};
```

The initial state should be the zero state and the transform state should be similar with the [basic tutorial](basic_tutorial.md). To get the robot, we clone the robot we created in the previous step:

```cpp
    std::shared_ptr<robot_dart::Robot> get_robot() const
    {
        std::shared_ptr<robot_dart::Robot> simulated_robot = global::global_robot->clone();

        return simulated_robot;
    }
```

To get the state of the robot we use some DART functions:

```cpp
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

//...
    Eigen::VectorXd get_state(const base_t::robot_t& robot, bool full) const
    {
        return get_robot_state(robot, full);
    }
//...
```

We add some extras to the simulation:

```cpp
    void add_extra_to_simu(base_t::robot_simu_t& simu) const
    {
        // Change gravity (as can be seen if you read the SKEL file)
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
```

Lastly, we need to define what type of actuators we want to use and the simulation time-step (i.e., not the control rate, but the physics engine time-step). In our case we set a very small simulation time-step (i.e., 0.001) to have a stable simulation and use `FORCE` actuators: i.e., torque-controlled joints.

```cpp
struct Params {
    // ...

    struct dart_system {
        BO_PARAM(double, sim_step, 0.001);
    };

    struct dart_policy_control {
        BO_PARAM(dart::dynamics::Joint::ActuatorType, joint_type, dart::dynamics::Joint::FORCE);
    };

    // ...
};
```

#### Learning the reward function

If we want to learn the reward function from data (both for DART-based and ODE-based scenarios), we need to do the following:

- Create a Gaussian process for learning the reward function:

```cpp
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
    // other code in global namespace ...

    using kernel_t = limbo::kernel::SquaredExpARD<RewardParams>;
    using mean_t = limbo::mean::Data<Params>;
    using GP_t = limbo::model::GP<RewardParams, kernel_t, mean_t, blackdrops::model::gp::KernelLFOpt<RewardParams>>;

    GP_t reward_gp(4, 1);
}
```

- Create a reward function struct for the actual immediate reward (this would be unknown to the algorithm):

```cpp
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
        double dee = (eef - global::goal).norm();

        return -dee;
    }
};
```

- Create a reward function to be used for the policy optimization:

```cpp
struct RewardFunction {
    double operator()(const Eigen::VectorXd& from_state, const Eigen::VectorXd& action, const Eigen::VectorXd& to_state, bool certain = false) const
    {
        Eigen::VectorXd mu;
        double s;
        std::tie(mu, s) = global::reward_gp.query(to_state);
        if (certain || !Params::opt_cmaes::handle_uncertainty())
            return mu(0);

        return gaussian_rand(mu(0), std::sqrt(s));
    }
};
```

- The system struct needs to override the `execute(const Policy& policy, const Reward& world, double T, std::vector<double>& R, bool display)` member function:

```cpp
template <typename Policy, typename Reward>
    std::vector<std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd>> execute(const Policy& policy, const Reward& world, double T, std::vector<double>& R, bool display = true)
    {
        // Run the policy on the system collecting the real reward samples
        ActualReward actual_reward;
        std::vector<std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd>> ret = blackdrops::system::DARTSystem<Params, PolicyControl>::execute(policy, actual_reward, T, R, display);

        // Add new samples to the reward GP
        std::vector<Eigen::VectorXd> states = this->get_last_states();

        for (size_t i = 0; i < R.size(); i++) {
            double r = R[i];
            Eigen::VectorXd final = states[i + 1];
            global::reward_gp.add_sample(final, limbo::tools::make_vector(r));
        }

        // Optimize the hyper-parameters of the reward GP
        global::reward_gp.optimize_hyperparams();
        std::cout << "Learned the new reward function..." << std::endl;

        return ret;
    }
```

#### Defining random and learning episodes

Around 2 random trials and 15 learning episodes should be enough to learn the task in most of the cases.

#### Compiling and running your scenario

**Using the provided scripts**

This requires that you have installed everything using the scripts.

You should now do the following:

- `cd /path/to/repo/root` **(this is very important as the scripts assume that you are in the root of the repo)**
- `./scripts/configure.sh` (if you have already done this, it is not needed)
- `./scripts/compile.sh`

If there's no error, you should be able to run your scenario:

- `source ./scripts/paths.sh` (this should be done only once for each terminal --- it should be run from the root of the repo)
- `./deps/limbo/build/exp/blackdrops/src/tutorials/dart_reacher2d_graphic -n 10 -m 5000 -e 1 -r 5 -b 1 -u`

You should now watch this simple robot trying to reach the target location. The task is considered solved for cumulative rewards >= -1. One minor remark is that this scenario can take quite some time between each episode (depending on your CPU capabilities).

For more detailed explanation of the command line arguments run: `./deps/limbo/build/exp/blackdrops/src/tutorials/dart_reacher2d_graphic -h`. If you do not have OpenSceneGraph installed, use `dart_reacher2d_simu` instead to run the experiment without graphics.

**For advanced users**

If you have used the advanced installation procedure, then you should do the following:

- `cd /path/to/limbo/folder`
- `./waf configure --exp blackdrops`
- `./waf --exp blackdrops -j4`

If there's no error, you should be able to run your scenario: `./build/exp/blackdrops/src/tutorials/dart_reacher2d_graphic -n 10 -m 5000 -e 1 -r 5 -b 1 -u`

### Where to put the files of my new DART-based scenario

When you want to create a new DART-based scenario, you should copy the `templates/dart_template.cpp` file into `src/dart/` folder, modify it (look for the `TO-CHANGE` parts in the code) and then compile using the instructions above. If you want to create an ODE-based scenario, then look at the [basic tutorial](basic_tutorial.md). If you require more fine tuned compilation of your program (e.g., link/include more libraries), then please make an issue and we will help you.