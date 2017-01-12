#ifndef MEDROPS_SF_NN_POLICY_HPP
#define MEDROPS_SF_NN_POLICY_HPP

#define EIGEN3_ENABLED
#include "nn2/mlp.hpp"

namespace medrops {

    template <typename Params, typename Model>
    struct SFNNPolicy {

        // using nn_t = medrops::MLP<medrops::NNLayer<medrops::Neuron<medrops::AfTanh>, medrops::PfSum>, medrops::NNLayer<medrops::Neuron<medrops::AfTanh>, medrops::PfSum>>;
        using nn_t = nn::Mlp<nn::Neuron<nn::PfWSum<>, nn::AfTanhNoBias<>>, nn::Connection<double, double>>;

        SFNNPolicy()
        {
            _boundary = Params::medrops::boundary();
            _random = false;
            _nn = std::make_shared<nn_t>(
                Params::nn_policy::state_dim(),
                Params::nn_policy::hidden_neurons(),
                Params::action_dim());
            _nn->init();
            _params = Eigen::VectorXd::Zero(_nn->get_nb_connections());
        }

        void normalize(const Model& model)
        {
            Eigen::MatrixXd data = model.samples();
            Eigen::MatrixXd samples = data.block(0, 0, data.rows(), data.cols() - 1);
            _means = samples.colwise().mean().transpose();
            _sigmas = Eigen::colwise_sig(samples).array().transpose();

            Eigen::VectorXd pl = Eigen::percentile(samples.array().abs(), 5);
            Eigen::VectorXd ph = Eigen::percentile(samples.array().abs(), 95);
            _limits = pl.array().max(ph.array());

#ifdef INTACT
            _limits << 16.138, 9.88254, 14.7047, 0.996735, 0.993532;
#endif
        }

        Eigen::VectorXd next(const Eigen::VectorXd& state) const
        {
            if (_random || _params.size() == 0) {
                return Params::nn_policy::max_u() * (limbo::tools::random_vector(Params::action_dim()).array() * 2 - 1.0);
            }

            Eigen::VectorXd nstate = state;
            // nstate.segment(0, 3) = ((state-_means).array() / (_sigmas*3).array()).segment(0, 3);
            nstate.segment(0, 3) = nstate.segment(0, 3).array() / _limits.segment(0, 3).array();

            std::vector<double> inputs(Params::nn_policy::state_dim());
            Eigen::VectorXd::Map(inputs.data(), inputs.size()) = nstate;

            _nn->step(inputs);
            _nn->step(inputs);

            std::vector<double> outputs = _nn->get_outf();
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
            if (_boundary == 0)
                _params = params;
            else
                _params = params.array() * 2.0 * _boundary - _boundary;
            _random = false;
            std::vector<double> weights(params.size());
            Eigen::VectorXd::Map(weights.data(), weights.size()) = params;
            _nn->set_all_weights(weights);
            _nn->init();
        }

        Eigen::VectorXd params(bool as_is = false) const
        {
            if (_random || _params.size() == 0)
                return limbo::tools::random_vector(_nn->get_nb_connections());
            if (_boundary == 0 || as_is)
                return _params;
            return (_params.array() + _boundary) / (_boundary * 2.0);
        }

        std::shared_ptr<nn_t> _nn;
        Eigen::VectorXd _params;
        bool _random;
        Model* _model;

        Eigen::VectorXd _means;
        Eigen::MatrixXd _sigmas;
        Eigen::VectorXd _limits;

        double _boundary;
    };
}

#endif
