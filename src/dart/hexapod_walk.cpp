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
#define MEAN
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

#include <hexapod_controller/hexapod_controller_simple.hpp>

#include <utils/utils.hpp>

struct Params {
#ifdef GRAPHIC
    struct graphics : robot_dart::defaults::graphics {
    };
#endif

    struct blackdrops : public ::blackdrops::defaults::blackdrops {
        BO_PARAM(size_t, action_dim, 18);
        BO_PARAM(size_t, model_input_dim, 49);
        BO_PARAM(size_t, model_pred_dim, 49);
        BO_PARAM(double, dt, 0.1);
        BO_PARAM(double, T, 4.0);
        BO_PARAM(double, boundary, 1.0);
        BO_DYN_PARAM(bool, verbose);
    };

    struct dart_system {
        BO_PARAM(double, sim_step, 0.01);
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

    struct fake_policy {
        BO_PARAM(size_t, state_dim, Params::blackdrops::model_input_dim());
        BO_PARAM(size_t, action_dim, Params::blackdrops::action_dim());
    };
};

struct FakePolicy {

    FakePolicy() { _random = false; }

    Eigen::VectorXd next(const Eigen::VectorXd& state) const
    {
        if (_random || _params.size() == 0) {
            Eigen::VectorXd act = (limbo::tools::random_vector(PolicyParams::fake_policy::action_dim()).array() * 2 - 1.0);
            return act;
        }

        auto angles = _simple_controller.pos(state(48));
        Eigen::VectorXd act = Eigen::VectorXd::Map(angles.data(), angles.size());

        return act;
    }

    void set_random_policy()
    {
        _random = true;
    }

    bool random() const
    {
        return _random;
    }

    void set_params(const Eigen::VectorXd& params)
    {
        _random = false;
        _params = params;

        std::vector<double> ctrl_params(36, 0.0);
        Eigen::VectorXd::Map(ctrl_params.data(), ctrl_params.size()) = (params.array() + 1.0) / 2.0;
        _simple_controller.set_parameters(ctrl_params);
    }

    Eigen::VectorXd params() const
    {
        if (_random || _params.size() == 0)
            return limbo::tools::random_vector(36);
        return _params;
    }

    Eigen::VectorXd _params;
    bool _random;
    hexapod_controller::HexapodControllerSimple _simple_controller;
};

namespace global {
    std::shared_ptr<robot_dart::Robot> global_robot;

    Eigen::VectorXd goal(3);

    std::vector<dart::dynamics::SkeletonPtr> available_robots;
    std::vector<bool> used_robots;
    std::mutex robot_mutex;
    int n_robots = 100;
} // namespace global

dart::dynamics::SkeletonPtr get_available_robot()
{
    std::lock_guard<std::mutex> lock(global::robot_mutex);
    for (int i = 0; i < global::n_robots; i++) {
        // std::lock_guard<std::mutex> lock(global::robot_mutex);
        if (global::used_robots[i] == false) {
            // std::lock_guard<std::mutex> lock(global::robot_mutex);
            // std::cout << "Giving robot '" << i << "'" << std::endl;
            global::used_robots[i] = true;
            return global::available_robots[i];
        }
    }

    return nullptr;
}

bool release_robot(const dart::dynamics::SkeletonPtr& robot)
{
    std::lock_guard<std::mutex> lock(global::robot_mutex);
    for (int i = 0; i < global::n_robots; i++) {
        // std::lock_guard<std::mutex> lock(global::robot_mutex);
        if (global::available_robots[i] == robot) {
            // std::lock_guard<std::mutex> lock(global::robot_mutex);
            // std::cout << "Releasing robot '" << i << "'" << std::endl;
            global::used_robots[i] = false;
            // global::recreating_robots[i] = true;
            global::available_robots[i]->clearExternalForces();
            global::available_robots[i]->clearInternalForces();
            global::available_robots[i]->clearConstraintImpulses();
            global::available_robots[i]->resetPositions();
            global::available_robots[i]->resetVelocities();
            global::available_robots[i]->resetAccelerations();
            global::available_robots[i]->resetGeneralizedForces();
            global::available_robots[i]->resetCommands();
            return true;
        }
    }

    return false;
}

Eigen::VectorXd get_robot_state(const std::shared_ptr<robot_dart::Robot>& robot)
{
    Eigen::VectorXd vels = robot->skeleton()->getVelocities();
    Eigen::VectorXd pos = robot->skeleton()->getPositions();
    int size = vels.size() + pos.size();

    Eigen::VectorXd state(size);
    state.head(vels.size()) = vels;
    state.segment(vels.size(), pos.size()) = pos;

    return state;
}

struct PolicyControl : public robot_dart::RobotControl {
public:
    using robot_t = std::shared_ptr<robot_dart::Robot>;

    PolicyControl() {}
    PolicyControl(const std::vector<double>& ctrl, robot_t robot)
        : robot_dart::RobotControl(ctrl, robot)
    {
        size_t _start_dof = 0;
        if (!_robot->fixed_to_world()) {
            _start_dof = 6;
        }
        _p = Eigen::VectorXd::Zero(_dof);
        _target_positions = _robot->skeleton()->getPositions();

        std::vector<size_t> indices;
        std::vector<dart::dynamics::Joint::ActuatorType> types;
        for (size_t i = _start_dof; i < _dof; i++) {
            _p(i) = 1.0;
            auto j = _robot->skeleton()->getDof(i)->getJoint();
            indices.push_back(_robot->skeleton()->getIndexOf(j));
            types.push_back(Params::dart_policy_control::joint_type());
        }
        _robot->set_actuator_types(indices, types);

        _prev_time = 0.0;
        _t = 0.0;

        _policy.set_params(Eigen::VectorXd::Map(ctrl.data(), ctrl.size()));

        _states.clear();
        _coms.clear();
        _first = true;
    }

    void update(double t)
    {
        _t = t;
        set_commands();
    }

    void set_commands()
    {
        double dt = Params::blackdrops::dt();

        if (_first || (_t - _prev_time - dt) > -Params::dart_system::sim_step() / 2.0) {
            Eigen::VectorXd q = this->get_state(_robot);
            q.conservativeResize(q.size() + 1);
            q.tail(1) = limbo::tools::make_vector(_t);

            Eigen::VectorXd commands = _policy.next(q);
            // std::vector<double> ctrl_params(36, 0.0);
            // Eigen::VectorXd::Map(ctrl_params.data(), ctrl_params.size()) = (commands.array() + 1.0) / 2.0;
            // // std::cout << "Setting: " << ctrl_params.size() << " " << commands.size() << std::endl;
            // _simple_controller.set_parameters(ctrl_params);
            // std::cout << "Set: " << _simple_controller.parameters().size() << std::endl;
            for (int i = 0; i < commands.size(); i++)
                _target_positions(i + 6) = ((i % 3 == 1) ? 1.0 : -1.0) * commands[i];

            _states.push_back(q);
            _coms.push_back(commands);

            _prev_time = _t;
            _first = false;
        }

        Eigen::VectorXd q_err = _target_positions - _robot->skeleton()->getPositions();

        double gain = 1.0 / (M_PI * _robot->skeleton()->getTimeStep());
        Eigen::VectorXd vel = q_err * gain;
        vel = vel.cwiseProduct(_p);
        assert(_dof == (size_t)vel.size());
        _robot->skeleton()->setCommands(vel);
        _prev_commands = vel;
    }

    std::vector<Eigen::VectorXd> get_states() const
    {
        return _states;
    }

    std::vector<Eigen::VectorXd> get_noiseless_states() const
    {
        return _states;
    }

    std::vector<Eigen::VectorXd> get_commands() const
    {
        return _coms;
    }

    Eigen::VectorXd get_state(const robot_t& robot) const
    {
        return get_robot_state(robot);
    }

    void set_transform_state(std::function<Eigen::VectorXd(const Eigen::VectorXd&)> func) {}

    void set_noise_function(std::function<Eigen::VectorXd(const Eigen::VectorXd&)> func) {}

    void set_policy_function(std::function<Eigen::VectorXd(const Eigen::VectorXd&)> func) {}

protected:
    double _prev_time;
    double _t;
    bool _first;
    Eigen::VectorXd _prev_commands;
    Eigen::VectorXd _p, _target_positions;
    FakePolicy _policy;
    std::vector<Eigen::VectorXd> _coms;
    std::vector<Eigen::VectorXd> _states;
};

struct Hexapod : public blackdrops::system::DARTSystem<Params, PolicyControl, blackdrops::RolloutInfo> {
    using base_t = blackdrops::system::DARTSystem<Params, PolicyControl, blackdrops::RolloutInfo>;

    Eigen::VectorXd init_state() const
    {
        Eigen::VectorXd state = Eigen::VectorXd::Zero(Params::blackdrops::model_pred_dim());
        state(29) = 0.2;
        return state;
    }

    Eigen::VectorXd transform_state(const Eigen::VectorXd& original_state) const
    {
        return original_state;
    }

    std::shared_ptr<robot_dart::Robot> get_robot() const
    {
        std::shared_ptr<robot_dart::Robot> simulated_robot = global::global_robot->clone();
        // simulated_robot->fix_to_world();
        simulated_robot->set_position_enforced(true);
        simulated_robot->skeleton()->setPosition(5, 0.2);

        // CAUTION: We cannot alter shapes in cloned objects, because they are not cloned

        return simulated_robot;
    }

    void add_extra_to_simu(base_t::robot_simu_t& simu, const blackdrops::RolloutInfo& info) const
    {
        // add floor
        simu.add_floor();

#ifdef GRAPHIC
        Eigen::Vector3d pos = {0., 3.5, 2.};
        Eigen::Vector3d look_at = {0.75, 0., 0.};
        simu.graphics()->fixed_camera(pos, look_at);

        // Add init position marker
        Eigen::Vector6d init_pose = Eigen::Vector6d::Zero();
        init_pose.tail(3) = this->init_state().segment(27, 3);
        // pose, dims, type, mass, color, name
        simu.add_ellipsoid(init_pose, {0.1, 0.1, 0.1}, "fixed", 1., dart::Color::Green(1.0), "init_marker");
        // remove collisions from init marker
        simu.world()->getSkeleton("init_marker")->getRootBodyNode()->setCollidable(false);
#endif
    }

    template <typename Policy, typename Reward>
    std::vector<std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd>> execute(const Policy& policy, const Reward& world, double T, std::vector<double>& R, bool display = true)
    {
        blackdrops::RolloutInfo info;

        std::vector<std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd>> ret = base_t::execute(policy, world, T, R, display, &info);

        if (display)
            std::cout << "last state: " << this->get_last_states().back().segment(27, 3).transpose() << std::endl;

        return ret;
    }

    template <typename Policy, typename Model, typename Reward>
    void execute_dummy(const Policy& policy, const Model& model, const Reward& world, double T, std::vector<double>& R, bool display = true)
    {
        base_t::execute_dummy(policy, model, world, T, R, display);

        if (display)
            std::cout << "last dummy state: " << this->get_last_dummy_states().back().segment(27, 3).transpose() << std::endl;
    }
};

struct RewardFunction {
    double operator()(const blackdrops::RolloutInfo& info, const Eigen::VectorXd& from_state, const Eigen::VectorXd& action, const Eigen::VectorXd& to_state, bool certain = false) const
    {
        // TO-DO: Maybe stop when safety limits are reached
        Eigen::Vector3d rot;
        rot << to_state(24), to_state(25), to_state(26);
        auto rot_mat = dart::math::expMapRot(rot);
        Eigen::Vector3d z_axis = {0.0, 0.0, 1.0};
        Eigen::Vector3d robot_z_axis = rot_mat * z_axis;
        double z_angle = std::atan2((z_axis.cross(robot_z_axis)).norm(), z_axis.dot(robot_z_axis));
        if (std::abs(z_angle) >= dart::math::constants<double>::half_pi())
            return -10.0;

        if (std::abs(to_state(28)) > 0.5)
            return -Params::blackdrops::dt();
        return to_state(27) * Params::blackdrops::dt();
    }
};

struct PolicyControlSimple : public robot_dart::RobotControl {
public:
    using robot_t = std::shared_ptr<robot_dart::Robot>;

    PolicyControlSimple() {}
    PolicyControlSimple(const std::vector<double>& ctrl, robot_t robot)
        : robot_dart::RobotControl(ctrl, robot)
    {
        size_t _start_dof = 0;
        if (!_robot->fixed_to_world()) {
            _start_dof = 6;
        }
        _p = Eigen::VectorXd::Zero(_dof);
        _target_positions = _robot->skeleton()->getPositions();

        std::vector<size_t> indices;
        std::vector<dart::dynamics::Joint::ActuatorType> types;
        for (size_t i = _start_dof; i < _dof; i++) {
            _p(i) = 1.0;
            auto j = _robot->skeleton()->getDof(i)->getJoint();
            indices.push_back(_robot->skeleton()->getIndexOf(j));
            types.push_back(Params::dart_policy_control::joint_type());
        }
        _robot->set_actuator_types(indices, types);

        _prev_time = 0.0;
        _t = 0.0;

        for (size_t i = 0; i < ctrl.size(); i++)
            _target_positions(i + 6) = ((i % 3 == 1) ? 1.0 : -1.0) * ctrl[i];

        _states.clear();
        _first = true;
    }

    void update(double t)
    {
        _t = t;
        set_commands();
    }

    void set_commands()
    {
        double dt = Params::blackdrops::dt();

        if (_first || (_t - _prev_time - dt) > -Params::dart_system::sim_step() / 2.0) {
            Eigen::VectorXd q = this->get_state(_robot);
            q.conservativeResize(q.size() + 1);
            q.tail(1) = limbo::tools::make_vector(_t);
            _states.push_back(q);

            _prev_time = _t;
            _first = false;
        }

        Eigen::VectorXd q_err = _target_positions - _robot->skeleton()->getPositions();

        double gain = 1.0 / (M_PI * _robot->skeleton()->getTimeStep());
        Eigen::VectorXd vel = q_err * gain;
        vel = vel.cwiseProduct(_p);
        assert(_dof == (size_t)vel.size());
        _robot->skeleton()->setCommands(vel);
        _prev_commands = vel;
    }

    std::vector<Eigen::VectorXd> get_states() const
    {
        return _states;
    }

    Eigen::VectorXd get_state(const robot_t& robot) const
    {
        return get_robot_state(robot);
    }

protected:
    double _prev_time;
    double _t;
    bool _first;
    Eigen::VectorXd _prev_commands;
    Eigen::VectorXd _p, _target_positions;
    std::vector<Eigen::VectorXd> _states;
    hexapod_controller::HexapodControllerSimple _simple_controller;
};

struct MeanFunc {
    using robot_simu_t = robot_dart::RobotDARTSimu<robot_dart::robot_control<PolicyControlSimple>>;

    MeanFunc(int dim_out = 1) {}

    template <typename GP>
    Eigen::VectorXd operator()(const Eigen::VectorXd& v, const GP& gp) const
    {
        double dt = Params::blackdrops::dt();

        Eigen::VectorXd ctrl = v.tail(18);
        std::vector<double> params(18, 0.0);
        Eigen::VectorXd::Map(params.data(), params.size()) = ctrl; //(ctrl.array() + 1.0) / 2.0;
        Eigen::VectorXd state = v.head(49);
        double t = state.tail(1)(0);

        for (int i = 0; i < state.size(); i++) {
            if (std::isnan(state(i)) || std::isinf(state(i))) {
                std::cerr << "Got NaN value! Returning zero instead!" << std::endl;
                return Eigen::VectorXd::Zero(state.size());
            }
        }

        auto rob = get_available_robot();
        while (rob == nullptr)
            rob = get_available_robot();

        auto simulated_robot = std::make_shared<robot_dart::Robot>(rob);

        simulated_robot->set_position_enforced(true);
        simulated_robot->skeleton()->setVelocities(state.head(24));
        simulated_robot->skeleton()->setPositions(state.segment(24, 24));

        robot_simu_t simu(params, simulated_robot);
        simu.add_floor();
        // simulation step different from sampling rate -- we need a stable simulation
        simu.set_step(Params::dart_system::sim_step());
        simu.world()->setTime(t);

        simu.run(dt + Params::dart_system::sim_step());

        // if (simu.controller().get_states().size() == 0)
        //     std::cout << "ERROR: " << state.transpose() << std::endl;

        Eigen::VectorXd final = simu.controller().get_states().back();

        release_robot(rob);

        return (final - state);
    }
};

void init_simu(const std::string& robot_file, int broken_leg = -1)
{
    global::global_robot = std::make_shared<robot_dart::Robot>(robot_dart::Robot(robot_file, {}, "hexapod", true));

    if (broken_leg >= 0 && broken_leg <= 5) {
        std::cout << "Shortened leg: " << broken_leg << std::endl;
        // small bug in DART forces us to do it here in global robot
        // shorten leg in "real" robot
        // std::string leg_bd_name = "leg_4_3";
        // std::string leg_bd_name = "leg_0_3";
        std::string leg_bd_name = "leg_" + std::to_string(broken_leg) + "_3";
        auto bd = global::global_robot->skeleton()->getBodyNode(leg_bd_name);
        bd->setMass(bd->getMass() / 2.0);
        auto nodes = bd->getShapeNodes();

        for (auto node : nodes) {
            Eigen::Vector3d tr = node->getRelativeTranslation();
            tr(1) = tr(1) / 2.0;
            node->setRelativeTranslation(tr);
            auto s = node->getShape();
            if (s->getType() == "BoxShape") {
                auto b = (dart::dynamics::BoxShape*)s.get();
                Eigen::Vector3d size = b->getSize();
                size(2) = size(2) / 2.0;
                b->setSize(size);
            }
            else if (s->getType() == "CylinderShape") {
                auto b = (dart::dynamics::CylinderShape*)s.get();
                b->setHeight(b->getHeight() / 2.0);
            }
        }
    }

    // Quick hack for not proper cloning of shapes in DART
    auto sim_robot = std::make_shared<robot_dart::Robot>(robot_dart::Robot(robot_file, {}, "hexapod", true));

    global::available_robots.clear();
    global::used_robots.clear();

    for (int i = 0; i < global::n_robots; i++) {
        global::available_robots.push_back(sim_robot->skeleton()->clone());
        global::used_robots.push_back(false);
    }
}

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
    int broken_leg = -1;
    std::string policy_file = "";

    namespace po = boost::program_options;
    po::options_description desc("Command line arguments");
    // clang-format off
    desc.add_options()("help,h", "Prints this help message")
                      ("max_evals,m", po::value<int>(), "Max function evaluations to optimize the policy.")
                      ("tolerance,t", po::value<double>(), "Maximum tolerance to continue optimizing the function.")
                      ("restarts,r", po::value<int>(), "Max number of restarts to use during optimization.")
                      ("elitism,e", po::value<int>(), "Elitism mode to use [0 to 3].")
                      ("uncertainty,u", po::bool_switch(&uncertainty)->default_value(false), "Enable uncertainty handling.")
                      ("threads,d", po::value<int>(), "Max number of threads used by TBB")
                      ("damage,p", po::value<int>(), "Leg to be shortened [0-5]")
                      ("verbose,v", po::bool_switch(&verbose)->default_value(false), "Enable verbose mode.")
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

        if (vm.count("threads")) {
            threads = vm["threads"].as<int>();
        }

        if (vm.count("damage")) {
            broken_leg = vm["damage"].as<int>();
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

    Params::opt_cmaes::set_lbound(-Params::blackdrops::boundary());
    Params::opt_cmaes::set_ubound(Params::blackdrops::boundary());

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

    init_simu(std::string(RESPATH) + "/URDF/pexod.urdf", broken_leg);

    using policy_opt_t = limbo::opt::Cmaes<Params>;

    using kernel_t = limbo::kernel::SquaredExpARD<Params>;
#ifndef MEAN
    using mean_t = limbo::mean::Constant<Params>;
#else
    using mean_t = MeanFunc;
#endif

    using GP_t = blackdrops::model::MultiGP<Params, limbo::model::GP, kernel_t, mean_t, blackdrops::model::multi_gp::MultiGPParallelLFOpt<Params, blackdrops::model::gp::KernelLFOpt<Params>>>;

    using MGP_t = blackdrops::GPModel<Params, GP_t>;

    blackdrops::BlackDROPS<Params, MGP_t, Hexapod, FakePolicy, policy_opt_t, RewardFunction> hexa_system;

    hexa_system.learn(1, 10, true, policy_file);

    return 0;
}