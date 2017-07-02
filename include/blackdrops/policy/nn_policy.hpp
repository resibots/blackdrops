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
