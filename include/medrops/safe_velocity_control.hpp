#ifndef SAFE_VELOCITY_CONTROL
#define SAFE_VELOCITY_CONTROL

#include <iostream>
#include <map>
#include <unordered_set>
#include <cmath> // for sin/cos, computing direct kinematic model
#include <dynamixel/dynamixel.hpp>

using namespace dynamixel;
using namespace controllers;
using namespace servos;
using namespace instructions;
using namespace protocols;

namespace dynamixel {
    class SafeVelocityControl {
    public:
        typedef typename Protocol1::id_t id_t;
        SafeVelocityControl(
            const std::string& usb_serial_port,
            std::unordered_set<Protocol1::id_t> selected_servos,
            const std::map<id_t, double>& min_angles,
            const std::map<id_t, double>& max_angles,
            const std::map<id_t, double>& max_velocities,
            double min_height)
            : _serial_interface(usb_serial_port, B1000000, 0.02),
              _min_angles(min_angles),
              _max_angles(max_angles),
              _max_velocities(max_velocities),
              _min_height(min_height)
        {
            _servos = auto_detect_map<Protocol1>(_serial_interface);
            // List detected servos
            std::cout << "Servos detected (" << _servos.size() << "):" << std::endl;
            for (auto servo : _servos) {
                std::cout << (int)servo.first
                          << "\t" << servo.second->model_name()
                          << std::endl;
            }
            // remove servos that are not in the _dynamixel_map (i.e. that are not used)
            for (auto servo_it = _servos.begin(); servo_it != _servos.end();) {
                typename std::unordered_set<Protocol1::id_t>::iterator dynamixel_iterator
                    = selected_servos.find((*servo_it).second->id());
                // if the actuator's name is not in the set, remove it
                if (dynamixel_iterator == selected_servos.end())
                    servo_it = _servos.erase(servo_it);
                else
                    ++servo_it;
            }
            // enable all actuators
            StatusPacket<Protocol1> status;
            for (auto servo : _servos) {
                _serial_interface.send(
                    servo.second->set_torque_enable(1));
                _serial_interface.recv(status);
            }
        }

        void go_to_target(const std::vector<double>& target, double threshold = 1e-3)
        {
            // move to target position
            double time_step = 0.05;

            std::map<dynamixel::protocols::Protocol1::id_t, double> velocities;
            velocities[1] = 0.0;
            velocities[2] = 0.0;
            velocities[3] = 0.0;
            velocities[4] = 0.0;

            std::vector<double> q = joint_angles();
            for (size_t i = 0; i < 4; i++)
                q[i] = q[i] - M_PI;
            std::vector<double> q_err(4, 0.0);
            for (size_t i = 0; i < 4; i++)
                q_err[i] = target[i] - q[i];
            double derr = -std::numeric_limits<double>::max();
            for (size_t i = 0; i < 4; i++) {
                if (std::abs(q_err[i]) > derr)
                    derr = std::abs(q_err[i]);
            }
            //std::accumulate(q_err.begin(), q_err.end(), 0.0, [](double a, double b) { return a*a+b*b; });

            auto prev_time = std::chrono::steady_clock::now();
            while (derr > threshold) {
                std::chrono::duration<double> elapsed_time = std::chrono::steady_clock::now() - prev_time;
                if (elapsed_time.count() <= 1e-5 || elapsed_time.count() >= time_step) {
                    prev_time = std::chrono::steady_clock::now();

                    q = joint_angles();
                    for (size_t i = 0; i < 4; i++)
                        q[i] = q[i] - M_PI;
                    for (size_t i = 0; i < 4; i++)
                        q_err[i] = target[i] - q[i];

                    double gain = 1.0 / (M_PI * time_step);

                    for (size_t i = 0; i < 4; i++) {
                        if (std::abs(q_err[i]) > threshold) {
                            velocities[i + 1] = q_err[i] * gain;
                            if (velocities[i + 1] > 1.0)
                                velocities[i + 1] = 1.0;
                            if (velocities[i + 1] < -1.0)
                                velocities[i + 1] = -1.0;
                        }
                        else
                            velocities[i + 1] = 0.0;
                    }

                    // derr = std::accumulate(q_err.begin(), q_err.end(), 0.0, [](double a, double b) { return a*a+b*b; });
                    derr = -std::numeric_limits<double>::max();
                    for (size_t i = 0; i < 4; i++) {
                        if (std::abs(q_err[i]) > derr)
                            derr = std::abs(q_err[i]);
                    }

                    // Send commands
                    velocity_command(velocities, true);
                }
            }

            // Zero velocity commands
            velocities[1] = 0.0;
            velocities[2] = 0.0;
            velocities[3] = 0.0;
            velocities[4] = 0.0;

            velocity_command(velocities);
        }

        void init_position()
        {
            std::vector<double> zero_pos = {0, 0, 0, 0};
            go_to_target(zero_pos);
        }

        /**
            The main method of this class. Stops all actuators if any one is beyond
            set joint limits.
            @return true if and only if the joints are outside the limits
        **/
        bool enforce_joint_limits()
        {
            std::map<id_t, double> angles = joint_angles_map();
            if (joint_limit_reached(angles) || height_limit_reached(angles)) {
                std::map<id_t, double> vels = angles;
                for (int i = 0; i < 4; i++)
                    vels[i + 1] = 0.0;
                velocity_command(vels);
                //    tell client software that we entered the safety mode
                return true;
            }
            else
                return false;
        }

        // /** Send a random velocity to each actuator
        // **/
        // void random_velocity_command()
        // {
        //     StatusPacket<Protocol1> status;
        //     for (auto servo : _servos) {
        //         double rand_vel
        //             = (std::rand() * 2.0 / RAND_MAX - 1) * _max_velocities.at(servo.first);
        //         std::cout << (int)servo.first << " -> " << rand_vel << "\t";
        //         _serial_interface.send((servo.second)->reg_goal_speed_angle(rand_vel, servos::cst::wheel));
        //         _serial_interface.recv(status);
        //     }
        //     std::cout << std::endl;
        //     _serial_interface.send(Action<Protocol1>(Protocol1::broadcast_id));
        // }

        /** Send velocity orders to all the actuators.
            @param vector of velocities, in ascending order of actuator ID;
        **/
        void velocity_command(const std::map<id_t, double>& velocities, bool ignore_height = false)
        {
            // check that we get the right number of torques
            if (velocities.size() != _servos.size()) {
                std::stringstream message;
                message << "velocity_command: Expecting " << _servos.size()
                        << " velocities and recieving " << velocities.size() << " instead.";
                throw errors::Error(message.str());
            }
            std::map<id_t, double> angles = joint_angles_map();
            std::map<id_t, double> vels;

            for (auto servo : _servos) {
                // if (-1 > velocities.at(servo.first) || velocities.at(servo.first) > 1)
                //     throw errors::Error("velocity_command: Velocity has to be between -1 and 1.");
                double vel = velocities.at(servo.first);
                if (_min_angles[servo.first] >= angles[servo.first])
                    vel = std::max(0.0, vel);
                // above max angle
                if (angles[servo.first] >= _max_angles[servo.first])
                    vel = std::min(0.0, vel);

                vels[servo.first] = vel;
            }

            // Lookahead
            double h_prev = height_reached(angles);

            if (h_prev <= _min_height && !ignore_height) {
                double dt = 0.05;
                for(auto angle : angles)
                {
                    angles.at(angle.first) = angle.second + vels[angle.first]*dt;
                }

                double h = height_reached(angles);
                if ((h-h_prev) <= 0.0)
                {
                    for (auto servo : _servos)
                      vels[servo.first] = 0.0;
                }
            }


            StatusPacket<Protocol1> status;
            for (auto servo : _servos) {
                _serial_interface.send((servo.second)->reg_goal_speed_angle(vels[servo.first], servos::cst::wheel));
                _serial_interface.recv(status);
            }
            _serial_interface.send(Action<Protocol1>(Protocol1::broadcast_id));
            // std::cout << std::endl;
        }

        /** Query the actuators for their current angles
            @return map from ids to doubles, one for each actuator
        **/
        std::map<id_t, double> joint_angles_map()
        {
            std::map<id_t, double> angles;
            StatusPacket<Protocol1> status;
            for (auto servo : _servos) {
                // request current position
                _serial_interface.send(
                    servo.second->get_present_position_angle());
                _serial_interface.recv(status);
                // parse response to get the position
                if (status.valid()) {
                    angles[servo.first]
                        = servo.second->parse_present_position_angle(status);
                }
                else {
                    std::stringstream message;
                    message << (int)servo.first << " did not answer to the request for "
                            << "its position";
                    throw errors::Error(message.str());
                }
            }
            return angles;
        }

        /** Query the actuators for their current angles
            @return vector of doubles, one for each actuator, in the same order
                as the list of servos
        **/
        std::vector<double> joint_angles()
        {
            std::vector<double> angles;
            StatusPacket<Protocol1> status;
            for (auto servo : _servos) {
                // request current position
                _serial_interface.send(
                    servo.second->get_present_position_angle());
                _serial_interface.recv(status);
                // parse response to get the position
                if (status.valid()) {
                    angles.push_back(
                        servo.second->parse_present_position_angle(status));
                }
                else {
                    std::stringstream message;
                    message << (int)servo.first << " did not answer to the request for "
                            << "its position";
                    throw errors::Error(message.str());
                }
            }
            return angles;
        }

        /** Forward kinematics of the arm
            @param angle positions of each joint
            @return 3D position of the end-effector
        **/
        std::vector<double> get_eef(const std::vector<double>& q)
        {
            std::vector<double> eef(3, 0.0);

            double a = 0.21, b = 0.208, c = 0.16, h = 0.176;

            // x-position
            eef[0] = std::cos(q[0]) * (a * std::sin(q[1]) + b * std::sin(q[1] + q[2]) + c * std::sin(q[1] + q[2] + q[3]));
            // y-position
            eef[1] = std::sin(q[0]) * (a * std::sin(q[1]) + b * std::sin(q[1] + q[2]) + c * std::sin(q[1] + q[2] + q[3]));
            // z-position (height of base[h] + eef)
            eef[2] = h + a * std::cos(q[1]) + b * std::cos(q[1] + q[2]) + c * std::cos(q[1] + q[2] + q[3]);

            return eef;
        }

        void set_min_angles(const std::map<id_t, double>& min_angles)
        {
            _min_angles = min_angles;
        }

        void set_max_angles(const std::map<id_t, double>& max_angles)
        {
            _max_angles = max_angles;
        }

        void set_max_velocities(const std::map<id_t, double>& max_velocities)
        {
            _max_velocities = max_velocities;
        }

        double min_height()
        {
            return _min_height;
        }

        void set_min_height(double new_height)
        {
            _min_height = new_height;
        }

    protected:
        /** Check whether one of the actuator reached a joint limit.
            The angles vector is modified so that it can be given as a position
            command to put back all joints within the joint limits.
            @param angles current joint angles for each actuator; this method
                moves any value outside of the bounds back within these bounds.
            @return true if and only if one or more joints are outside the limits
        **/
        bool joint_limit_reached(std::map<id_t, double>& angles)
        {
            if (angles.size() != _servos.size()) {
                std::cout << "Error: the size of the angles vector does not "
                          << "match with the number of servos.";
            } // throw an exception

            for (auto angle : angles) {
                // bellow min angle
                if (_min_angles[angle.first] >= angle.second)
                    return true;
                // above max angle
                if (angle.second >= _max_angles[angle.first])
                    return true;
            }

            return false;
        }

        /** Check whether the end effector is lower than the given height.
            This method is specific to our robotic arm based on Dynamixel Pros.
            @param angles current joint angles for each actuator
            @param reference the height bellow which the arm is not allowed to go
            @return true if and only if the end effector'h height is lower than reference
        **/
        bool height_limit_reached(const std::map<id_t, double>& angles)
        {
            std::vector<double> q;
            for (int i = 0; i < 4; i++)
                q.push_back(angles.at(i + 1) - M_PI);
            // get only z-position of the end-effector
            double z = get_eef(q)[2];
            return (z <= _min_height);
        }

        double height_reached(const std::map<id_t, double>& angles)
        {
            std::vector<double> q;
            for (int i = 0; i < 4; i++)
                q.push_back(angles.at(i + 1) - M_PI);
            // get only z-position of the end-effector
            double z = get_eef(q)[2];
            return z;
        }

        Usb2Dynamixel _serial_interface;
        std::map<id_t, std::shared_ptr<BaseServo<Protocol1>>> _servos;
        std::map<id_t, double> _min_angles;
        std::map<id_t, double> _max_angles;
        std::map<id_t, double> _max_velocities;
        double _min_height;
    };
}
#endif
