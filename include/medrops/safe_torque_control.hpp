#ifndef SAFE_TORQUE_CONTROL
#define SAFE_TORQUE_CONTROL

#include <iostream>
#include <map>
#include <unordered_set>

#include <dynamixel/dynamixel.hpp>

using namespace dynamixel;
using namespace controllers;
using namespace servos;
using namespace instructions;
using namespace protocols;

namespace dynamixel {
    class SafeTorqueControl {
    public:
        typedef typename Protocol2::id_t id_t;

        SafeTorqueControl(
            const std::string& usb_serial_port,
            std::unordered_set<Protocol2::id_t> selected_servos,
            const std::map<id_t, double>& min_angles,
            const std::map<id_t, double>& max_angles,
            const std::map<id_t, double>& max_torques)
            : _serial_interface(usb_serial_port, B115200, 0.005),
              _min_angles(min_angles),
              _max_angles(max_angles),
              _max_torques(max_torques)
        {
            _servos = auto_detect_map<Protocol2>(_serial_interface);

            // List detected servos
            std::cout << "Servos detected (" << _servos.size() << "):" << std::endl;
            for (auto servo : _servos) {
                std::cout << (int)servo.first
                          << "\t" << servo.second->model_name()
                          << std::endl;
            }

            // remove servos that are not in the _dynamixel_map (i.e. that are not used)
            for (auto servo_it = _servos.begin(); servo_it != _servos.end();) {
                typename std::unordered_set<Protocol2::id_t>::iterator dynamixel_iterator
                    = selected_servos.find((*servo_it).second->id());
                // if the actuator's name is not in the set, remove it
                if (dynamixel_iterator == selected_servos.end())
                    servo_it = _servos.erase(servo_it);
                else
                    ++servo_it;
            }

            // enable all actuators
            StatusPacket<Protocol2> status;
            for (auto servo : _servos) {
                _serial_interface.send(
                    servo.second->set_torque_enable(1));
                _serial_interface.recv(status);
            }
        }

        void init_position()
        {
            // move all joints to neutral position : middle between max and min angles
            StatusPacket<Protocol2> status;

            for (auto servo : _servos) {
                // move to middle between min and max angles
                _serial_interface.send(
                    servo.second->set_goal_position_angle(
                        (_max_angles[servo.first] + _min_angles[servo.first]) / 2));
                _serial_interface.recv(status);
            }
        }

        void reset_to_position_control()
        {
            StatusPacket<Protocol2> status;
            // First pass: torque goal to 0 and switch to position control
            for (auto servo : _servos) {
                _serial_interface.send(servo.second->set_goal_torque(0));
                _serial_interface.recv(status);
                //    disable torque output
                _serial_interface.send(
                    servo.second->set_torque_enable(0));
                _serial_interface.recv(status);
                //    move to position control
                _serial_interface.send(
                    servo.second->set_operating_mode(3));
                _serial_interface.recv(status);
            }

            // Second pass: reboot and enable actuators
            for (auto servo : _servos) {
                // reboot to a clean state
                _serial_interface.send(Reboot<Protocol2>(servo.first));

                // wait for the servo to respond to a ping request
                do {
                    _serial_interface.send(servo.second->ping());
                } while (!_serial_interface.recv(status));

                // enable torque again
                _serial_interface.send(
                    servo.second->set_torque_enable(1));
                _serial_interface.recv(status);
            }
        }

        void set_torque_control_mode()
        {
            // switch to torque control
            change_control_mode();
        }

        /**
            The main method of this class. Stops all actuators if any one is beyond
            set joint limits.

            @return true if and only if the joints are outside the limits
        **/
        bool enforce_joint_limits()
        {
            std::map<id_t, double> angles = joint_angles_map();

            if (joint_limit_reached(angles)) {
                // container for the servo's responses
                StatusPacket<Protocol2> status;

                // test lower timeout ?
                // auto recv_timeout = _serial_interface.recv_timeout();
                // _serial_interface.set_recv_timeout(0.002);

                // First pass: torque goal to 0 and switch to position control
                for (auto servo : _servos) {
                    _serial_interface.send(servo.second->set_goal_torque(0));
                    _serial_interface.recv(status);
                    //    disable torque output
                    _serial_interface.send(
                        servo.second->set_torque_enable(0));
                    _serial_interface.recv(status);
                    //    move to position control
                    _serial_interface.send(
                        servo.second->set_operating_mode(3));
                    _serial_interface.recv(status);
                }

                // Second pass: reboot and enable actuators
                for (auto servo : _servos) {
                    // reboot to a clean state
                    _serial_interface.send(Reboot<Protocol2>(servo.first));

                    std::cout << (int)servo.first << ": reboot command sent ";

                    // wait for the servo to respond to a ping request
                    do {
                        std::cout << ".";
                        _serial_interface.send(servo.second->ping());
                    } while (!_serial_interface.recv(status));
                    std::cout << std::endl;

                    // enable torque again
                    _serial_interface.send(
                        servo.second->set_torque_enable(1));
                    _serial_interface.recv(status);

                    // set goal position to either lateset one or the min/max
                    // if it went overboard
                    _serial_interface.send(
                        servo.second->set_goal_position_angle(angles[servo.first]));
                    _serial_interface.recv(status);
                }

                // _serial_interface.set_recv_timeout(recv_timeout);

                //    tell client software that we entered the safety mode
                return true;
            }
            else
                return false;
        }

        /** Send a random torque to each actuator, within a percentage of the
            max torque.

            @param ratio of the torque that can be used (between 0 and 1)
        **/
        void random_torque_command(double limit_torque_ratio)
        {
            StatusPacket<Protocol2> status;

            for (auto servo : _servos) {
                double rand_torque
                    = (std::rand() * 2.0 / RAND_MAX - 1)
                    * _max_torques.at(servo.first) * limit_torque_ratio;
                std::cout << (int)servo.first << " -> " << rand_torque << "\t";
                _serial_interface.send((servo.second)->reg_goal_torque(rand_torque));
                _serial_interface.recv(status);
            }
            std::cout << std::endl;

            _serial_interface.send(Action<Protocol2>(Protocol2::broadcast_id));
        }

        /** Send torque orders to all the actuators.

            @param vector of torques, in ascending order of actuator ID;
                value is between 0 and 1;
                there MUST be as many elements in the vector as there are actuators in torque control mode
        **/
        void torque_command(const std::map<id_t, double>& torques)
        {
            // check that we get the right number of torques
            if (torques.size() != _servos.size()) {
                std::stringstream message;
                message << "torque_command: Expecting " << _servos.size()
                        << " torques and recieving " << torques.size() << " instead.";
                throw errors::Error(message.str());
            }

            StatusPacket<Protocol2> status;
            for (auto servo : _servos) {
                if (-1 > torques.at(servo.first) || torques.at(servo.first) > 1)
                    throw errors::Error("torque_command: Torque has to be between -1 and 1.");

                double torque = torques.at(servo.first) * _max_torques.at(servo.first);
                // std::cout << (int)servo.first << " -> " << torque << "\t";
                _serial_interface.send((servo.second)->reg_goal_torque(torque));
                _serial_interface.recv(status);
            }
            _serial_interface.send(Action<Protocol2>(Protocol2::broadcast_id));
            // std::cout << std::endl;
        }

        /** Querry the actuators for their current angles

            @return map from ids to doubles, one for each actuator
        **/
        std::map<id_t, double> joint_angles_map()
        {
            std::map<id_t, double> angles;

            StatusPacket<Protocol2> status;

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

        /** Querry the actuators for joint position and joint speed, returning
            them concatenated.

            @return vector of joint speeds followed by vector of joint positions
                in ascending order of joint ID.
        **/
        std::vector<double> concatenated_joint_state()
        {
            std::vector<double> state;

            StatusPacket<Protocol2> status;

            // First, we get speed information
            for (auto servo : _servos) {
                // request current speed
                _serial_interface.send(
                    servo.second->get_present_speed());
                _serial_interface.recv(status);

                // parse response to get the speed
                if (status.valid()) {
                    state.push_back(
                        servo.second->parse_joint_speed(status));
                }
                else {
                    std::stringstream message;
                    message << (int)servo.first << " did not answer to the "
                            << "request for its speed";
                    throw errors::Error(message.str());
                }
            }

            // Then the joint positions
            for (auto servo : _servos) {
                // request current position
                _serial_interface.send(
                    servo.second->get_present_position_angle());
                _serial_interface.recv(status);

                // parse response to get the position
                if (status.valid()) {
                    state.push_back(
                        servo.second->parse_present_position_angle(status));
                }
                else {
                    std::stringstream message;
                    message << (int)servo.first << " did not answer to the request for "
                            << "its position";
                    throw errors::Error(message.str());
                }
            }

            return state;
        }

        /** Querry the actuators for their current angles

            @return vector of doubles, one for each actuator, in the same order
                as the list of servos
        **/
        std::vector<double> joint_angles()
        {
            std::vector<double> angles;

            StatusPacket<Protocol2> status;

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

        /** Querry the actuators for their current angular speeds

            @return vector of doubles, one for each actuator, in the same order as
                returned by ids
        **/
        std::vector<double> joint_speeds()
        {
            std::vector<double> angles;

            StatusPacket<Protocol2> status;

            for (auto servo : _servos) {
                // request current speed
                _serial_interface.send(
                    servo.second->get_present_speed());
                _serial_interface.recv(status);

                // parse response to get the speed
                if (status.valid()) {
                    angles.push_back(
                        servo.second->parse_joint_speed(status));
                }
                else {
                    std::stringstream message;
                    message << (int)servo.first << " did not answer to the "
                            << "request for its speed";
                    throw errors::Error(message.str());
                }
            }

            return angles;
        }

        void set_min_angles(const std::map<id_t, double>& min_angles)
        {
            _min_angles = min_angles;
        }

        void set_max_angles(const std::map<id_t, double>& max_angles)
        {
            _max_angles = max_angles;
        }

        void set_max_torques(const std::map<id_t, double>& max_torques)
        {
            _max_torques = max_torques;
        }

    protected:
        /** Change the control mode (a.k.a. operating mode) of the actuators.

            CAUTION: This will need to disable all actuators in the process.

            @param operating_mode desired operating mode, as defined in the documentation:
                0: torque control
                1: velocity control
                3: position control
                4: extended position control (multi-turn)
        **/
        void change_control_mode(int operating_mode = 0)
        {
            // container for the servo's responses
            StatusPacket<Protocol2> status;

            // move to torque control
            std::cout << "CAUTION: temporarily disabling torque for all actuators!" << std::endl;
            for (auto servo : _servos) {
                // disable torque
                _serial_interface.send(
                    servo.second->set_torque_enable(0));
                _serial_interface.recv(status);
                // move to torque control
                _serial_interface.send(
                    servo.second->set_operating_mode(operating_mode));
                _serial_interface.recv(status);
                // enable torque
                _serial_interface.send(
                    servo.second->set_torque_enable(1));
                _serial_interface.recv(status);
            }
            std::cout << "\ttorque enabled again" << std::endl;
        }

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

            bool reached_limit = false;

            for (auto angle : angles) {
                // bellow min angle
                if (_min_angles[angle.first] >= angle.second) {
                    angles[angle.first] = _min_angles[angle.first];
                    reached_limit = true;
                }
                // above max angle
                else if (angle.second >= _max_angles[angle.first]) {
                    angles[angle.first] = _max_angles[angle.first];
                    reached_limit = true;
                }
            }
            return reached_limit;
        }

        Usb2Dynamixel _serial_interface;
        std::map<id_t, std::shared_ptr<BaseServo<Protocol2>>> _servos;
        std::map<id_t, double> _min_angles;
        std::map<id_t, double> _max_angles;
        std::map<id_t, double> _max_torques;
    };
}

#endif
