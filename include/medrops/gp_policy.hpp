#ifndef MEDROPS_GP_POLICY_HPP
#define MEDROPS_GP_POLICY_HPP

#include <limbo/tools/random_generator.hpp>
#include <Eigen/Core>
#include <limbo/tools/macros.hpp>

namespace medrops {

    template <typename Params>
    struct GPPolicy {

        GPPolicy() { _random = false; }

        Eigen::VectorXd next(const Eigen::VectorXd& state) const
        {
            if (_random || _params.size() == 0) {
                return Params::linear_policy::max_u() * (limbo::tools::random_vector(Params::action_dim()).array() * 2 - 1); //return max action
            }

            Eigen::VectorXd A = basisFunction(state);
            Eigen::VectorXd act = A.dot(_alphas);
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
            size_t N = Params::linear_policy::state_dim();
            _params = params;
            for (int i=0; i<20 ; i++)
                for (int j=0; j<N; j++)
                {
                    _thetas[i](j) =params(i*N + j);
                }
            _alphas = params.tail(N);//Eigen::MatrixXd::Map(_params.data(),N,N).transpose();
            _random = false;
        }

        Eigen::VectorXd params() const
        {
            if (_random || _params.size() == 0)
                return limbo::tools::random_vector((Params::linear_policy::state_dim() + 1) * Params::action_dim());
            return _params;
        }


        double exp_kernel(const Eigen::VectorXd v1, const Eigen::VectorXd v2) const
        {
            double _l = Params::gp_policy::l();
            double _sigma = 10;
            return std::exp(-(1 / (2 * std::pow(_l, 2))) * std::pow((v1 - v2).norm(), 2));
        }

        Eigen::VectorXd basisFunction(const Eigen::VectorXd& state) const
        {
            int N = Params::gp_policy::virtual_observations();//20; //total parameters 20*state.size()+20
            Eigen::VectorXd result(N,0.0);
            for (int i=0; i<N; i++ )
            {
                result(i) = exp_kernel(state,_thetas[i]);
            }
            return result;
        }

        Eigen::VectorXd _params;
        std::vector<Eigen::VectorXd> _thetas; //pseudo samples
        Eigen::VectorXd _alphas; //covarience_matrix x pseudo observations
        bool _random;
    };
}

#endif
