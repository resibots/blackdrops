//| Copyright Inria July 2017
//| This project has received funding from the European Research Council (ERC) under
//| the European Union's Horizon 2020 research and innovation programme (grant
//| agreement No 637972) - see http://www.resibots.eu
//|
//| Contributor(s):
//|   - Konstantinos Chatzilygeroudis (konstantinos.chatzilygeroudis@inria.fr)
//|   - Rituraj Kaushik (rituraj.kaushik@inria.fr)
//|   - Roberto Rama (bertoski@gmail.com)
//|
//| This software is the implementation of the Black-DROPS algorithm, which is
//| a model-based policy search algorithm with the following main properties:
//|   - uses Gaussian processes (GPs) to model the dynamics of the robot/system
//|   - takes into account the uncertainty of the dynamical model when
//|                                                      searching for a policy
//|   - is data-efficient or sample-efficient; i.e., it requires very small
//|     interaction time with the system to find a working policy (e.g.,
//|     around 16-20 seconds to learn a policy for the cart-pole swing up task)
//|   - when several cores are available, it can be faster than analytical
//|                                                    approaches (e.g., PILCO)
//|   - it imposes no constraints on the type of the reward function (it can
//|                                                  also be learned from data)
//|   - it imposes no constraints on the type of the policy representation
//|     (any parameterized policy can be used --- e.g., dynamic movement
//|                                              primitives or neural networks)
//|
//| Main repository: http://github.com/resibots/blackdrops
//| Preprint: https://arxiv.org/abs/1703.07261
//|
//| This software is governed by the CeCILL-C license under French law and
//| abiding by the rules of distribution of free software.  You can  use,
//| modify and/ or redistribute the software under the terms of the CeCILL-C
//| license as circulated by CEA, CNRS and INRIA at the following URL
//| "http://www.cecill.info".
//|
//| As a counterpart to the access to the source code and  rights to copy,
//| modify and redistribute granted by the license, users are provided only
//| with a limited warranty  and the software's author,  the holder of the
//| economic rights,  and the successive licensors  have only  limited
//| liability.
//|
//| In this respect, the user's attention is drawn to the risks associated
//| with loading,  using,  modifying and/or developing or reproducing the
//| software by the user in light of its specific status of free software,
//| that may mean  that it is complicated to manipulate,  and  that  also
//| therefore means  that it is reserved for developers  and  experienced
//| professionals having in-depth computer knowledge. Users are therefore
//| encouraged to load and test the software's suitability as regards their
//| requirements in conditions enabling the security of their systems and/or
//| data to be ensured and,  more generally, to use and operate it in the
//| same conditions as regards security.
//|
//| The fact that you are presently reading this means that you have had
//| knowledge of the CeCILL-C license and that you accept its terms.
//|
#ifndef BLACKDROPS_POLICY_GP_POLICY_HPP
#define BLACKDROPS_POLICY_GP_POLICY_HPP

#include <Eigen/Core>
#include <limbo/limbo.hpp>
#include <limbo/tools.hpp>
#include <limbo/tools/macros.hpp>
#include <limbo/tools/random_generator.hpp>

namespace blackdrops {
    namespace defaults {
        struct gp_policy {
            BO_PARAM(double, max_u, 10.0); //max action
            BO_PARAM(double, pseudo_samples, 10);
            BO_PARAM(double, noise, 0.01);
        };
    } // namespace defaults

    namespace policy {
        template <typename Params>
        struct GPPolicy {
            using kernel_t = limbo::kernel::SquaredExpARD<Params>;
            using mean_t = limbo::mean::Data<Params>;
            using gp_t = limbo::model::GP<Params, kernel_t, mean_t>;

            GPPolicy()
            {
                _boundary = Params::blackdrops::boundary();
                _random = false;
                _sdim = Params::gp_policy::state_dim();
                _adim = Params::gp_policy::action_dim();
                _ps = Params::gp_policy::pseudo_samples();
                _params = Eigen::VectorXd::Zero(_ps * _sdim + _adim * (_ps + _sdim));
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
                Eigen::VectorXd nstate = state;
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
                    Eigen::VectorXd ell = _params.segment(_ps * (_sdim + _adim) + j * _sdim, _sdim);
                    Eigen::VectorXd sigma_ell(ell.size() + 1);
                    sigma_ell.head(ell.size()) = ell;
                    sigma_ell.tail(1) = limbo::tools::make_vector(0.0);
                    ells.push_back(sigma_ell);

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

            std::vector<gp_t> _gp_policies;
        };
    } // namespace policy
} // namespace blackdrops
#endif
