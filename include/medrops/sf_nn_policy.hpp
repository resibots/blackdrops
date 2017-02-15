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
            _boundary = Params::medrops::boundary();
            _random = false;
            _nn = std::make_shared<nn_t>(
                Params::nn_policy::state_dim(),
                Params::nn_policy::hidden_neurons(),
                Params::action_dim());
            _nn->init();
            _params = Eigen::VectorXd::Zero(_nn->get_nb_connections());
            // _limits = Eigen::VectorXd(6);
            // _limits << 2.67992, 2.20705, 3.89703, 0.300936, 0.274408, 0.488859;
            // // _limits << 30.0, 30.0, 30.0, 3.14, 3.14, 3.14; 10.5124 23.5371 35.8726 3.14391 1.81071 1.83343
            // // _limits << 9.36686, 19.3844, 38.1972, 3.15046, 1.81577, 1.86595;
            // // _limits << 10.1127, 24.3294, 42.2179, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0;
            // _limits << 1.0, 1.0, 1.0, 1.0, 1.0, 1.0;
            _limits = Eigen::VectorXd::Constant(Params::nn_policy::state_dim(), 1.0);
        }

        template <typename Model>
        void normalize(const Model& model)
        {
            Eigen::MatrixXd data = model.samples();
            Eigen::MatrixXd samples = data.block(0, 0, data.rows(), Params::model_input_dim());
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
                Eigen::VectorXd act = (limbo::tools::random_vector(Params::action_dim()).array() * 2 - 1.0);
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

#endif
