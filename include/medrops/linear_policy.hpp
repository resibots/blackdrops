#ifndef MEDROPS_LINEAR_POLICY_HPP
#define MEDROPS_LINEAR_POLICY_HPP

#include <limbo/tools/random_generator.hpp>

namespace medrops {

    template <typename Params>
    struct LinearPolicy {

        LinearPolicy() { _random = false; }

        Eigen::VectorXd next(const Eigen::VectorXd& state) const
        {
            if (_random || _params.size() == 0) {
                Eigen::VectorXd act = (limbo::tools::random_vector(Params::action_dim()).array() * 2 - 1.0);
                for (int i = 0; i < act.size(); i++) {
                    act(i) = act(i) * Params::linear_policy::max_u(i);
                }
                return act;
            }

            Eigen::VectorXd act = _alpha * state + _constant;

            for (int i = 0; i < act.size(); i++) {
                act(i) = Params::linear_policy::max_u(i) * (9 * std::sin(act(i)) / 8.0 + std::sin(3 * act(i)) / 8.0);
            }

            return act;
        }

        template <typename Model>
        void normalize(const Model& model) {}

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
            size_t M = Params::action_dim();
            size_t N = Params::linear_policy::state_dim();

            _params = params;
            _alpha = Eigen::MatrixXd::Zero(M, N);
            _constant = Eigen::VectorXd::Zero(M);
            for (size_t i = 0; i < M; i++) {
                for (size_t j = 0; j < N; j++) {
                    _alpha(i, j) = params(i * M + j);
                }
            }

            for (size_t i = N * M; i < (N + 1) * M; i++)
                _constant(i - N * M) = params(i);

            _random = false;
        }

        Eigen::VectorXd params() const
        {
            if (_random || _params.size() == 0)
                return limbo::tools::random_vector((Params::linear_policy::state_dim() + 1) * Params::action_dim());
            return _params;
        }

        Eigen::MatrixXd _alpha;
        Eigen::VectorXd _constant, _params;
        bool _random;
    };
}

#endif
