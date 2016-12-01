#ifndef MEDROPS_SF_NN_POLICY_HPP
#define MEDROPS_SF_NN_POLICY_HPP

#define EIGEN3_ENABLED
#include "nn2/mlp.hpp"

namespace medrops {

    template <typename Params>
    struct SFNNPolicy {

        // using nn_t = medrops::MLP<medrops::NNLayer<medrops::Neuron<medrops::AfTanh>, medrops::PfSum>, medrops::NNLayer<medrops::Neuron<medrops::AfTanh>, medrops::PfSum>>;
        using nn_t = nn::Mlp<nn::Neuron<nn::PfWSum<>, nn::AfTanhNoBias<>>, nn::Connection<double, double>>;

        SFNNPolicy()
        {
            _random = false;
            _nn = std::make_shared<nn_t>(
                Params::nn_policy::state_dim(),
                Params::nn_policy::hidden_neurons(),
                Params::action_dim()
            );
            _nn->init();
            _params = Eigen::VectorXd::Zero(_nn->get_nb_connections());
        }

        Eigen::VectorXd next(const Eigen::VectorXd& state) const
        {
            if (_random || _params.size() == 0) {
                return Params::nn_policy::max_u() * (limbo::tools::random_vector(Params::action_dim()).array() * 2 - 1.0);
            }

            std::vector<double> inputs(Params::nn_policy::state_dim());
            Eigen::VectorXd::Map(inputs.data(), inputs.size()) = state;

            // for (size_t i = 0; i < inputs.size(); i++)
            //   std::cout << inputs[i] << " ";
            // std::cout << std::endl;

            _nn->step(inputs);
            _nn->step(inputs);

            std::vector<double> outputs = _nn->get_outf();
            Eigen::VectorXd act = Eigen::VectorXd::Map(outputs.data(), outputs.size());

            // std::vector<double> out_weights = _nn->get_all_weights();
            // Eigen::VectorXd weights = Eigen::VectorXd::Map(out_weights.data(), out_weights.size());
            // std::cout << "weights: " << weights.segment(0, 6).transpose() << std::endl;
            // std::cout << "action: " << act.transpose() << std::endl;

            act = act.unaryExpr([](double x) {
                return Params::nn_policy::max_u() * x;//(9 * std::sin(x) / 8.0 + std::sin(3 * x) / 8.0);
            });
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

        void set_params(const Eigen::VectorXd& params) {
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
                return limbo::tools::random_vector(_nn->get_nb_connections());
            return _params;
        }

        std::shared_ptr<nn_t> _nn;
        Eigen::VectorXd _params;
        bool _random;
    };
}

#endif
