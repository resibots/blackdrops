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
#ifndef BLACKDROPS_REWARD_GP_REWARD_HPP
#define BLACKDROPS_REWARD_GP_REWARD_HPP

#include <blackdrops/reward/reward.hpp>

#include <limbo/model/gp.hpp>

namespace blackdrops {

    struct reward_defaults {
        struct kernel : public limbo::defaults::kernel {
            BO_PARAM(bool, optimize_noise, true);
        };

        struct kernel_squared_exp_ard : public limbo::defaults::kernel_squared_exp_ard {
        };

        struct mean_constant {
            BO_PARAM(double, constant, 0.);
        };

        struct opt_rprop : public limbo::defaults::opt_rprop {
            BO_PARAM(int, iterations, 300);
            BO_PARAM(double, eps_stop, 1e-4);
        };
    };

    template <typename Params>
    using RewardGP = limbo::model::GP<Params, limbo::kernel::SquaredExpARD<Params>, limbo::mean::Constant<Params>, blackdrops::model::gp::KernelLFOpt<Params, limbo::opt::Rprop<Params>>>;

    namespace reward {

        template <typename MyReward, typename GP = RewardGP<reward_defaults>>
        struct GPReward : public Reward<MyReward> {
            template <typename RolloutInfo>
            double observe(const RolloutInfo& info, const Eigen::VectorXd& from_state, const Eigen::VectorXd& action, const Eigen::VectorXd& to_state, bool keep = true)
            {
                double val = (*static_cast<const MyReward*>(this))(info, from_state, action, to_state);

                if (keep) {
                    Eigen::VectorXd sample = static_cast<const MyReward*>(this)->get_sample(info, from_state, action, to_state);
                    _samples.push_back(sample);
                    _obs.push_back(limbo::tools::make_vector(val));
                }

                return val;
            }

            template <typename RolloutInfo>
            double query(const RolloutInfo& info, const Eigen::VectorXd& from_state, const Eigen::VectorXd& action, const Eigen::VectorXd& to_state) const
            {
                Eigen::VectorXd mu;
                double sigma;
                std::tie(mu, sigma) = _model.query(static_cast<const MyReward*>(this)->get_sample(info, from_state, action, to_state));

                return std::min(mu[0] + std::sqrt(sigma), std::max(mu[0] - std::sqrt(sigma), gaussian_rand(mu[0], sigma)));
            }

            bool learn()
            {
                _model.compute(_samples, _obs, false);
                _model.optimize_hyperparams();

                return true;
            }

            template <typename RolloutInfo>
            Eigen::VectorXd get_sample(const RolloutInfo& info, const Eigen::VectorXd& from_state, const Eigen::VectorXd& action, const Eigen::VectorXd& to_state) const
            {
                Eigen::VectorXd vec(to_state.size() + action.size());
                vec.head(to_state.size()) = to_state;
                vec.tail(action.size()) = action;

                return vec;
            }

        protected:
            std::vector<Eigen::VectorXd> _samples, _obs;
            GP _model;
        };
    } // namespace reward
} // namespace blackdrops

#endif