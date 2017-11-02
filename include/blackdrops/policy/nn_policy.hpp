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
#ifndef BLACKDROPS_POLICY_NN_POLICY_HPP
#define BLACKDROPS_POLICY_NN_POLICY_HPP

#define EIGEN3_ENABLED
#include "nn2/mlp.hpp"

namespace blackdrops {
    namespace policy {
        template <typename Params>
        struct NNPolicy {

            using nn_t = nn::Mlp<nn::Neuron<nn::PfWSum<>, nn::AfTanhNoBias<>>, nn::Connection<double, double>>;

            NNPolicy()
            {
                _boundary = Params::blackdrops::boundary();
                _random = false;
                _nn = std::make_shared<nn_t>(
                    Params::nn_policy::state_dim(),
                    Params::nn_policy::hidden_neurons(),
                    Params::nn_policy::action_dim());
                _nn->init();
                _params = Eigen::VectorXd::Zero(_nn->get_nb_connections());
                _limits = Eigen::VectorXd::Constant(Params::nn_policy::state_dim(), 1.0);

                // Get the limits
                for (int i = 0; i < _limits.size(); i++) {
                    _limits(i) = Params::nn_policy::limits(i);
                }

                std::vector<float> afs(_nn->get_nb_neurons(), Params::nn_policy::af());
                _nn->set_all_afparams(afs);
            }

            Eigen::VectorXd next(const Eigen::VectorXd& state) const
            {
                if (_random || _params.size() == 0) {
                    Eigen::VectorXd act = (limbo::tools::random_vector(Params::nn_policy::action_dim()).array() * 2 - 1.0);
                    for (int i = 0; i < act.size(); i++) {
                        act(i) = act(i) * Params::nn_policy::max_u(i);
                    }
                    return act;
                }

                Eigen::VectorXd nstate = state.array() / _limits.array(); //((state - _means).array() / (_sigmas * 3).array()); //state.array() / _limits.array();

                std::vector<double> inputs(Params::nn_policy::state_dim());
                Eigen::VectorXd::Map(inputs.data(), inputs.size()) = nstate;

                _nn->step(inputs);
                _nn->step(inputs);

                std::vector<double> outputs = _nn->get_outf();
                Eigen::VectorXd act = Eigen::VectorXd::Map(outputs.data(), outputs.size());

                for (int i = 0; i < act.size(); i++) {
                    act(i) = act(i) * Params::nn_policy::max_u(i);
                }
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
                _params = params;
                _random = false;
                std::vector<double> weights(params.size());
                Eigen::VectorXd::Map(weights.data(), weights.size()) = params;
                _nn->set_all_weights(weights);
                _nn->init();
            }

            Eigen::VectorXd params() const
            {
                if (_random || _params.size() == 0)
                    return limbo::tools::random_vector(_nn->get_nb_connections()).array() * 2.0 * _boundary - _boundary;
                return _params;
            }

            std::shared_ptr<nn_t> _nn;
            Eigen::VectorXd _params;
            bool _random;

            Eigen::VectorXd _means;
            Eigen::MatrixXd _sigmas;
            Eigen::VectorXd _limits;

            double _boundary;
        };
    }
}
#endif
