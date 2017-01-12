#ifndef MEDROPS_GP_POLICY_HPP
#define MEDROPS_GP_POLICY_HPP
#include <limbo/tools/random_generator.hpp>
#include <Eigen/Core>
#include <limbo/tools/macros.hpp>
#include <limbo/limbo.hpp>
#include <limbo/tools.hpp>

namespace medrops {
    namespace defaults {
        struct gp_policy_defaults {
            BO_PARAM(double, max_u, 10.0); //max action
            BO_PARAM(double, pseudo_samples, 10);
            BO_PARAM(double, noise, 1e-5);
        };
    }
    template <typename Params, typename Model>
    struct GPPolicy {
        using kernel_t = limbo::kernel::SquaredExpARD<Params>;
        using mean_t = limbo::mean::Constant<Params>;
        using gp_t = limbo::model::GP<Params, kernel_t, mean_t>;

        GPPolicy()
        {
            _random = false;
            _sdim = Params::nn_policy::state_dim();
            _ps = Params::gp_policy::pseudo_samples();
            _params = Eigen::VectorXd::Zero((_sdim + 1) * _ps + _sdim);
        }

        void normalize(const Model& model) {}

        Eigen::VectorXd next(const Eigen::VectorXd state) const
        {
            Eigen::VectorXd policy_params;
            policy_params = _params;

            if (_random || policy_params.size() == 0) {
                return Params::gp_policy::max_u() * (limbo::tools::random_vector(Params::action_dim()).array() * 2 - 1); //return random action
            }

            //---extract pseudo samples from parameters
            Eigen::VectorXd sample(_sdim);
            std::vector<Eigen::VectorXd> pseudo_samples;
            for (size_t i = 0; i < _ps; i++) {
                for (size_t j = 0; j < _sdim; j++) {
                    sample(j) = policy_params(i * _sdim + j);
                }
                pseudo_samples.push_back(sample);
            }

            //--- extract pseudo observations from parameres
            Eigen::VectorXd obs;
            std::vector<Eigen::VectorXd> pseudo_observations;
            obs = policy_params.segment(_sdim * _ps, _ps);
            for (int i = 0; i < obs.size(); i++) {
                Eigen::VectorXd temp = limbo::tools::make_vector(obs(i));
                pseudo_observations.push_back(temp);
            }

            //--- extract hyperparameters from parameters
            Eigen::VectorXd ells(_sdim);
            ells = policy_params.tail(_sdim);

            //-- instantiating gp policy
            gp_t gp_policy_obj(_sdim, 1);

            //--- set hyperparameter ells in the kernel.
            gp_policy_obj.kernel_function().set_h_params(ells);

            //--- Compute the gp
            Eigen::VectorXd noises = Eigen::VectorXd::Constant(_ps, Params::gp_policy::noise());
            gp_policy_obj.compute(pseudo_samples, pseudo_observations, noises); //TODO: Have to check the noises with actual PILCO

            //--- Query the GP with state
            Eigen::VectorXd action = gp_policy_obj.mu(state);
            action = action.unaryExpr([](double x) {return Params::gp_policy::max_u() * (9 * std::sin(x) / 8.0 + std::sin(3 * x) / 8.0); });

            return action;
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
            _random = false;
            _params = params;
        }

        Eigen::VectorXd params(bool as_is = false) const
        {
            if (_random || _params.size() == 0)
                return limbo::tools::random_vector((_sdim + 1) * _ps + _sdim); //TODO: set the proper bounds here
            return _params;
        }

    protected:
        size_t _sdim; //input dimension
        size_t _ps; //total observations
        Eigen::VectorXd _params;
        bool _random;
    };
}
#endif
