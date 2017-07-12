## Implemented Scenarios

In this page, we will briefly present all the already implemented scenarios and how to run/use them properly.

### Pendulum swing-up task

The classic pendulum swing-up task from the robot control literature with the following properties:

- Duration of 4 seconds for each episode
- Sampling/control rate at 10Hz (i.e., steps of 0.1 seconds)
- 1 random trial

#### How to run it

The recommended parameters to use are the following:

- 10000 maximum functions evaluations for CMA-ES
- 5 restarts for CMA-ES
- Enable elitism for CMA-ES
- 10 hidden neurons for the neural network policy
- [-5,5] the boundaries for the parameters of the policy

In short, you should run: `./deps/limbo/build/exp/blackdrops/src/classic_control/pendulum_simu -m 10000 -r 5 -n 10 -b 5 -e 1 -u`

If you want to try with the Gaussian process (GP) policy, you should run: `./deps/limbo/build/exp/blackdrops/src/classic_control/pendulum_simu_gppolicy -m 10000 -r 5 -b 1 -e 1 -u`

#### Results

<center>
<img src="../imgs/pendulum_rewards.png" width="400">
</center>

### Cart-pole swing-up task

The classic cart-pole swing-up task from the robot control literature with the following properties:

- Duration of 4 seconds for each episode
- Sampling/control rate at 10Hz (i.e., steps of 0.1 seconds)
- 1 random trial

#### How to run it

The recommended parameters to use are the following:

- 40000 maximum functions evaluations for CMA-ES
- 5 restarts for CMA-ES
- Enable elitism for CMA-ES
- 10 hidden neurons for the neural network policy
- [-5,5] the boundaries for the parameters of the policy

In short, you should run: `./deps/limbo/build/exp/blackdrops/src/classic_control/cartpole_simu -m 40000 -r 5 -n 10 -b 5 -e 1 -u`

If you want to try with the Gaussian process (GP) policy, you should run: `./deps/limbo/build/exp/blackdrops/src/classic_control/cartpole_simu_gppolicy -m 40000 -r 5 -b 1 -e 1 -u`

#### Results

<center>
<img src="../imgs/cartpole_rewards.png" width="400">
</center>

### Velocity-controlled arm reacher

This scenario was designed to replicate as close as possible the real robot scenario in the [Black-DROPS paper](https://arxiv.org/abs/1703.07261). A 4-DOF velocity-controlled arm has to find a policy in order for its end-effector to reach a desired target location. In addition, the immediate reward function is not known to the system/robot and has to learn it from data. In more detail:

- State space (4-D): [q0, q1, q2, q3]
- Action space (4-D): [dq0, dq1, dq2, dq3]
- The actuators take velocity commands, but operate in torque mode (i.e., they respect the maximum torques, maximum velocities and joint limits)
- Duration of 4 seconds for each episode
- Sampling/control rate at 10Hz (i.e., steps of 0.1 seconds)
- 2 random trials

#### How to run it

The recommended parameters to use are the following:

- 5000 maximum functions evaluations for CMA-ES
- 5 restarts for CMA-ES
- Enable elitism for CMA-ES
- 10 hidden neurons for the neural network policy
- [-1,1] the boundaries for the parameters of the policy

In short, you should run: `./deps/limbo/build/exp/blackdrops/src/dart/simple_arm_graphic -m 5000 -r 5 -n 10 -b 1 -e 1 -u`

If you want to run it without graphics: `./deps/limbo/build/exp/blackdrops/src/dart/simple_arm_simu -m 5000 -r 5 -n 10 -b 1 -e 1 -u`

#### Results

TO-DO: Add results