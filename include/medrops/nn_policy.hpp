#ifndef MEDROPS_NN_POLICY_HPP
#define MEDROPS_NN_POLICY_HPP

#include <nn2/mlp.hpp>

namespace medrops {

    template <typename Params>
    struct NNPolicy {

        using nn_t = nn::Mlp<nn::Neuron<nn::PfWSum<double>, nn::AfGaussian<double>>, nn::Connection<double, double>>;

        NNPolicy()
        {
            _random = false;
            size_t M = Params::action_dim();
            size_t N = Params::nn_policy::state_dim();
            size_t H = Params::nn_policy::hidden_neurons();

            _mlp = std::make_shared<nn_t>(N, H, M);
            _mlp->init();

            _params = Eigen::VectorXd::Zero(_mlp->get_nb_connections());
        }

        // NNPolicy(const NNPolicy& other)
        // {
        //     size_t M = Params::action_dim();
        //     size_t N = Params::nn_policy::state_dim();
        //     size_t H = Params::nn_policy::hidden_neurons();
        //
        //     _random = other._random;
        //     _mlp = std::make_shared<nn_t>(N, H, M);
        //     _mlp->init();
        //
        //     set_params(other._params);
        // }

        Eigen::VectorXd next(const Eigen::VectorXd& state) const
        {
            if (_random || _params.size() == 0) {
                return Params::nn_policy::max_u() * (limbo::tools::random_vector(Params::action_dim()).array() * 2 - 1.0);
            }

            std::vector<double> inputs(Params::nn_policy::state_dim());
            std::vector<double> outputs;
            Eigen::VectorXd::Map(inputs.data(), inputs.size()) = state;
            _mlp->step(inputs);
            _mlp->step(inputs);
            outputs = _mlp->get_outf();

            Eigen::VectorXd act = Eigen::VectorXd::Map(outputs.data(), outputs.size());

            act = act.unaryExpr([](double x) {
                return Params::linear_policy::max_u() * (9 * std::sin(x) / 8.0 + std::sin(3 * x) / 8.0);
            });
            return act;
        }

        void set_random_policy()
        {
            _random = true;
        }

        void set_params(const Eigen::VectorXd& params)
        {
            _params = params;
            _random = false;
            std::vector<double> weights(_mlp->get_nb_connections());
            Eigen::VectorXd::Map(weights.data(), weights.size()) = params;

            _mlp->set_all_weights(weights);
            _mlp->init();
        }

        Eigen::VectorXd params() const
        {
            if (_random || _params.size() == 0)
                return limbo::tools::random_vector(_mlp->get_nb_connections());
            return _params;
        }

        std::shared_ptr<nn_t> _mlp;
        Eigen::VectorXd _params;
        bool _random;
    };
}

#endif
