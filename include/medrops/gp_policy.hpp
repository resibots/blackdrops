#ifndef MEDROPS_GP_POLICY_HPP
#define MEDROPS_GP_POLICY_HPP

#include <limbo/tools/random_generator.hpp>
#include <Eigen/Core>
#include <limbo/tools/macros.hpp>
#include <limbo/limbo.hpp>

namespace medrops {

    template <typename Params>
    struct GPPolicy {

        GPPolicy()
        {
            _random = false;
             size_t sdim = Params::nn_policy::state_dim();
             size_t ps = Params::gp_policy::pseudo_samples();
             _params = Eigen::VectorXd::Zero(sdim*ps+ps);
        }

        Eigen::VectorXd next(const Eigen::VectorXd& state) const
        {
            if (_random || _params.size() == 0) {
                return Params::linear_policy::max_u() * (limbo::tools::random_vector(Params::action_dim()).array() * 2 - 1); //return max action
            }

            Eigen::VectorXd A = basisFunction(state);
            //std::cout<<"A = "<<A<<std::endl;
            //std::cout<<"_alphas = "<<_alphas<<std::endl;
            Eigen::VectorXd act = limbo::tools::make_vector(A.dot(_alphas));
            //std::cout<<act<<std::endl;
            act = act.unaryExpr([](double x) {
                return Params::gp_policy::max_u() * (9 * std::sin(x) / 8.0 + std::sin(3 * x) / 8.0);
            });
            //std::cout<<"ACTION = "<<act<<std::endl;
            return act;
            //return limbo::tools::make_vector();

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
            int M = Params::gp_policy::pseudo_samples();
            _params = params;
            _thetas = std::vector<Eigen::VectorXd>(M, Eigen::VectorXd(N));
            for (int i=0; i<M; i++)
                for (int j=0; j<N; j++)
                {
                    _thetas[i](j) =params(i*N + j);
                }
            _alphas = params.tail(M);//Eigen::MatrixXd::Map(_params.data(),N,N).transpose();
            _random = false;
        }

        Eigen::VectorXd params() const
        {
            if (_random || _params.size() == 0)
                // return limbo::tools::random_vector((Params::linear_policy::state_dim() + 1) * Params::action_dim());
                return limbo::tools::random_vector(Params::linear_policy::state_dim()*Params::gp_policy::pseudo_samples()+Params::gp_policy::pseudo_samples());
            return _params;
        }


        double exp_kernel(const Eigen::VectorXd v1, const Eigen::VectorXd v2) const
        {
            double _l = Params::gp_policy::l();
            //double ll= 1 / (2 * std::pow(_l, 2));
            //double diff = std::pow((v1 - v2).norm(), 2);
            //std::cout<<"ll = "<<ll<<std::endl;
            //std::cout<<"diff = "<<diff<<std::endl;
            return std::exp(-(1 / (2 * std::pow(_l, 2))) * std::pow((v1 - v2).norm(), 2));
        }

        Eigen::VectorXd basisFunction(const Eigen::VectorXd& state) const
        {
            int N = Params::gp_policy::pseudo_samples();//20; //total parameters 20*state.size()+20
            Eigen::VectorXd result(N);
            for (int i=0; i<N; i++ )
            {
                result(i) = exp_kernel(state,_thetas[i]);
                 //std::cout<<"state = "<<state<<std::endl;
                 //std::cout<<"_thetas = "<<_thetas[i]<<std::endl;
                // std::cout<<"Result = "<<result(i)<<std::endl;
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
