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
#ifndef UTILS_CMD_ARGS_HPP
#define UTILS_CMD_ARGS_HPP

#include <boost/program_options.hpp>

#include <limbo/tools/parallel.hpp>

namespace po = boost::program_options;

namespace utils {
    class CmdArgs {
    public:
        CmdArgs() : _verbose(false), _stochastic(false), _uncertainty(false), _threads(tbb::task_scheduler_init::automatic), _desc("Command line arguments") { _set_defaults(); }

        int parse(int argc, char** argv)
        {
            try {
                po::variables_map vm;
                po::store(po::parse_command_line(argc, argv, _desc), vm);
                if (vm.count("help")) {
                    std::cout << _desc << std::endl;
                    return 0;
                }

                po::notify(vm);

                if (vm.count("threads")) {
                    _threads = vm["threads"].as<int>();
                }
                if (vm.count("hidden_neurons")) {
                    int c = vm["hidden_neurons"].as<int>();
                    if (c < 1)
                        c = 1;
                    _neurons = c;
                }
                else {
                    _neurons = 5;
                }
                if (vm.count("pseudo_samples")) {
                    int c = vm["pseudo_samples"].as<int>();
                    if (c < 1)
                        c = 1;
                    _pseudo_samples = c;
                }
                else {
                    _pseudo_samples = 10;
                }
                if (vm.count("boundary")) {
                    double c = vm["boundary"].as<double>();
                    if (c < 0)
                        c = 0;
                    _boundary = c;
                }
                else {
                    _boundary = 0.;
                }

                // Cmaes parameters
                if (vm.count("max_evals")) {
                    int c = vm["max_evals"].as<int>();
                    _max_fun_evals = c;
                }
                else {
                    _max_fun_evals = -1;
                }
                if (vm.count("tolerance")) {
                    double c = vm["tolerance"].as<double>();
                    if (c < 0.)
                        c = 0.;
                    _fun_tolerance = c;
                }
                else {
                    _fun_tolerance = 1.;
                }
                if (vm.count("restarts")) {
                    int c = vm["restarts"].as<int>();
                    if (c < 1)
                        c = 1;
                    _restarts = c;
                }
                else {
                    _restarts = 1;
                }
                if (vm.count("elitism")) {
                    int c = vm["elitism"].as<int>();
                    if (c < 0 || c > 3)
                        c = 0;
                    _elitism = c;
                }
                else {
                    _elitism = 0;
                }
            }
            catch (po::error& e) {
                std::cerr << "[Exception caught while parsing command line arguments]: " << e.what() << std::endl;
                return 1;
            }

            return -1;
        }

        bool verbose() const { return _verbose; }
        bool stochastic() const { return _stochastic; }
        bool uncertainty() const { return _uncertainty; }

        int threads() const { return _threads; }
        int neurons() const { return _neurons; }
        int pseudo_samples() const { return _pseudo_samples; }
        int max_fun_evals() const { return _max_fun_evals; }
        int restarts() const { return _restarts; }
        int elitism() const { return _elitism; }

        double boundary() const { return _boundary; }
        double fun_tolerance() const { return _fun_tolerance; }

    protected:
        bool _verbose, _stochastic, _uncertainty;
        int _threads, _neurons, _pseudo_samples, _max_fun_evals, _restarts, _elitism;
        double _boundary, _fun_tolerance;

        po::options_description _desc;

        void _set_defaults()
        {
            // clang-format off
            _desc.add_options()("help,h", "Prints this help message")
                               ("hidden_neurons,n", po::value<int>(), "Number of hidden neurons in NN policy.")
                               ("pseudo_samples,p", po::value<int>(), "Number of pseudo samples in GP policy.")
                               ("boundary,b", po::value<double>(), "Boundary of the values during the optimization.")
                               ("max_evals,m", po::value<int>(), "Max function evaluations to optimize the policy.")
                               ("tolerance,t", po::value<double>(), "Maximum tolerance to continue optimizing the function.")
                               ("restarts,r", po::value<int>(), "Max number of restarts to use during optimization.")
                               ("elitism,e", po::value<int>(), "Elitism mode to use [0 to 3].")
                               ("uncertainty,u", po::bool_switch(&_uncertainty)->default_value(false), "Enable uncertainty handling in CMA-ES.")
                               ("stochastic,s", po::bool_switch(&_stochastic)->default_value(false), "Enable stochastic rollouts (i.e., not use the mean model).")
                               ("threads,d", po::value<int>(), "Max number of threads used by TBB")
                               ("verbose,v", po::bool_switch(&_verbose)->default_value(false), "Enable verbose mode.");
            // clang-format on
        }
    };
} // namespace utils

#endif