#ifndef MEDROPS_NN_POLICY_HPP
#define MEDROPS_NN_POLICY_HPP

#include <nn/nn.hpp>

namespace medrops {

    template <typename Params, typename Model>
    struct NNPolicy {

        using nn_t = medrops::MLP<medrops::NNLayer<medrops::Neuron<medrops::AfTanh>, medrops::PfSum>, medrops::NNLayer<medrops::Neuron<medrops::AfTanh>, medrops::PfSum>>;

        NNPolicy()
        {
            _random = false;
            size_t M = Params::action_dim();
            size_t N = Params::nn_policy::state_dim();
            size_t H = Params::nn_policy::hidden_neurons();

            _mlp = std::make_shared<nn_t>(N, std::vector<size_t>(1, H), M);

            _params = Eigen::VectorXd::Zero(_mlp->n_weights());
        }

        void normalize(const Model& model) {}

        Eigen::VectorXd next(const Eigen::VectorXd& state) const
        {
            if (_random || _params.size() == 0) {
                return Params::nn_policy::max_u() * (limbo::tools::random_vector(Params::action_dim()).array() * 2 - 1.0);
            }

            std::vector<double> inputs(Params::nn_policy::state_dim());
            std::vector<double> outputs;
            Eigen::VectorXd::Map(inputs.data(), inputs.size()) = state;

            outputs = _mlp->compute(inputs);

            Eigen::VectorXd act = Eigen::VectorXd::Map(outputs.data(), outputs.size());

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

        void set_params(const Eigen::VectorXd& params)
        {
            _params = params;
            _random = false;
            std::vector<double> weights(_mlp->n_weights());
            Eigen::VectorXd::Map(weights.data(), weights.size()) = params;

            _mlp->set_weights(weights);
        }

        Eigen::VectorXd params() const
        {
            if (_random || _params.size() == 0)
                return limbo::tools::random_vector(_mlp->n_weights());
            return _params;
        }

        std::shared_ptr<nn_t> _mlp;
        Eigen::VectorXd _params;
        bool _random;
    };
}

#endif
