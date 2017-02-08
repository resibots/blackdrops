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
            _adim = Params::action_dim();
            _ps = Params::gp_policy::pseudo_samples();
            _params = Eigen::VectorXd::Zero(_ps * _sdim + _adim * (_ps + _sdim));
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
            Eigen::VectorXd policy_params;
            policy_params = _params;

            if (_random || policy_params.size() == 0) {
                Eigen::VectorXd act = (limbo::tools::random_vector(Params::action_dim()).array() * 2 - 1.0);
                for (int i = 0; i < act.size(); i++) {
                    act(i) = act(i) * Params::gp_policy::max_u(i);
                }
                return act;
            }

            //---extract pseudo samples from parameters
            Eigen::VectorXd sample(_sdim);
            std::vector<Eigen::VectorXd> pseudo_samples;
            for (size_t i = 0; i < _ps; i++) {
                sample = policy_params.segment(i * _sdim, _sdim);
                pseudo_samples.push_back(sample);
            }

            //--- extract pseudo observations and hyperparameters from parameters
            Eigen::VectorXd obs;
            std::vector<std::vector<Eigen::VectorXd>> pseudo_observations;
            std::vector<Eigen::VectorXd> ells;
            for (size_t j = 0; j < _adim; j++) {
                //--- extract hyperparameters
                Eigen::VectorXd ell = policy_params.segment(_ps * (_sdim + _adim) + j * _sdim, _sdim);
                ells.push_back(ell);

                //--- extract pseudo observations
                obs = policy_params.segment(_sdim * _ps + j * _ps, _ps);
                pseudo_observations.push_back(std::vector<Eigen::VectorXd>());
                for (int i = 0; i < obs.size(); i++) {
                    Eigen::VectorXd temp = limbo::tools::make_vector(obs(i));
                    pseudo_observations[j].push_back(temp);
                }
            }

            //-- instantiating gp policy
            Eigen::VectorXd noises = Eigen::VectorXd::Constant(_ps, Params::gp_policy::noise());
            std::vector<gp_t> gp_policy(_adim, gp_t(_sdim, 1));
            tbb::parallel_for(size_t(0), _adim, size_t(1), [&](size_t i) {
                gp_policy[i].kernel_function().set_h_params(ells[i]);
                gp_policy[i].compute(pseudo_samples, pseudo_observations[i], noises);
            });

            //--- Query the GPs with state
            Eigen::VectorXd nstate = state.array() / _limits.array();
            Eigen::VectorXd action(_adim);
            for (size_t i = 0; i < _adim; i++) {
                Eigen::VectorXd a = gp_policy[i].mu(nstate);
                action(i) = a(0);
            }

            for (int i = 0; i < action.size(); i++) {
                action(i) = Params::gp_policy::max_u(i) * (9 * std::sin(action(i)) / 8.0 + std::sin(3 * action(i)) / 8.0);
            }

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

        Eigen::VectorXd params() const
        {
            if (_random || _params.size() == 0)
                return limbo::tools::random_vector(_ps * _sdim + _adim * (_ps + _sdim));
            return _params;
        }

    protected:
        size_t _sdim; //input dimension
        size_t _ps; //total observations
        size_t _adim; // action dimension
        Eigen::VectorXd _params;
        bool _random;
        double _boundary;

        Eigen::VectorXd _means;
        Eigen::MatrixXd _sigmas;
        Eigen::VectorXd _limits;
    };
}
#endif
