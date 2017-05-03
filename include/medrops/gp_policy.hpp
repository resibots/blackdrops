#ifndef MEDROPS_GP_POLICY_HPP
#define MEDROPS_GP_POLICY_HPP
#include <limbo/tools/random_generator.hpp>
#include <Eigen/Core>
#include <limbo/tools/macros.hpp>
#include <limbo/limbo.hpp>
#include <limbo/tools.hpp>

namespace medrops {
    namespace defaults {
        struct gp_policy {
            BO_PARAM(double, max_u, 10.0); //max action
            BO_PARAM(double, pseudo_samples, 10);
            BO_PARAM(double, noise, 0.01);
        };
    }

    template <typename Params>
    struct GPPolicy {
        using kernel_t = limbo::kernel::SquaredExpARD<Params>;
        using mean_t = limbo::mean::Data<Params>;
        using gp_t = limbo::model::GP<Params, kernel_t, mean_t>;

        GPPolicy()
        {
            _boundary = Params::medrops::boundary();
            _random = false;
            _sdim = Params::gp_policy::state_dim();
            _adim = Params::gp_policy::action_dim();
            _ps = Params::gp_policy::pseudo_samples();
            _params = Eigen::VectorXd::Zero(_ps * _sdim + _adim * (_ps + _sdim + 1));
            _limits = Eigen::VectorXd::Constant(Params::nn_policy::state_dim(), 1.0);
        }

        template <typename Model>
        void normalize(const Model& model)
        {
            _limits = model.limits().head(Params::nn_policy::state_dim());
        }

        Eigen::VectorXd next(const Eigen::VectorXd& state) const
        {
            if (_random || _params.size() == 0) {
                Eigen::VectorXd act = (limbo::tools::random_vector(_adim).array() * 2 - 1.0);
                for (int i = 0; i < act.size(); i++) {
                    act(i) = act(i) * Params::gp_policy::max_u(i);
                }
                return act;
            }

            //--- Query the GPs with state
            Eigen::VectorXd nstate = state.array() / _limits.array();
            Eigen::VectorXd action(_adim);
            tbb::parallel_for(size_t(0), _adim, size_t(1), [&](size_t i) {
                Eigen::VectorXd a = _gp_policies[i].mu(nstate);
                action(i) = Params::gp_policy::max_u(i) * (9.0 * std::sin(a(0)) / 8.0 + std::sin(3 * a(0)) / 8.0);
            });

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

            //---extract pseudo samples from parameters
            Eigen::VectorXd sample(_sdim);
            std::vector<Eigen::VectorXd> pseudo_samples;
            for (size_t i = 0; i < _ps; i++) {
                sample = _params.segment(i * _sdim, _sdim);
                pseudo_samples.push_back(sample);
            }

            //--- extract pseudo observations and hyperparameters from parameters
            Eigen::VectorXd obs;
            std::vector<std::vector<Eigen::VectorXd>> pseudo_observations;
            std::vector<Eigen::VectorXd> ells;
            for (size_t j = 0; j < _adim; j++) {
                //--- extract hyperparameters
                Eigen::VectorXd ell = _params.segment(_ps * (_sdim + _adim) + j * (_sdim + 1), _sdim + 1);
                ells.push_back(ell);

                //--- extract pseudo observations
                obs = _params.segment(_sdim * _ps + j * _ps, _ps);
                pseudo_observations.push_back(std::vector<Eigen::VectorXd>());
                for (int i = 0; i < obs.size(); i++) {
                    Eigen::VectorXd temp = limbo::tools::make_vector(obs(i));
                    pseudo_observations[j].push_back(temp);
                }
            }

            //-- instantiating gp policy
            _gp_policies.resize(_adim, gp_t(_sdim, 1));
            tbb::parallel_for(size_t(0), _adim, size_t(1), [&](size_t i) {
                _gp_policies[i].kernel_function().set_h_params(ells[i]);
                _gp_policies[i].compute(pseudo_samples, pseudo_observations[i]);
            });
        }

        Eigen::VectorXd params() const
        {
            if (_random || _params.size() == 0)
                return limbo::tools::random_vector(_ps * _sdim + _adim * (_ps + _sdim + 1));
            return _params;
        }

    protected:
        size_t _sdim; //input dimension
        size_t _ps; //total observations
        size_t _adim; // action dimension
        Eigen::VectorXd _params;
        bool _random;
        double _boundary;

        std::vector<gp_t> _gp_policies;

        Eigen::VectorXd _means;
        Eigen::MatrixXd _sigmas;
        Eigen::VectorXd _limits;
    };
}
#endif
