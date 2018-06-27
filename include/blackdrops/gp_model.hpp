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
#ifndef BLACKDROPS_GP_MODEL_HPP
#define BLACKDROPS_GP_MODEL_HPP

#include <Eigen/binary_matrix.hpp>

namespace blackdrops {

    template <typename Params, typename GP_t>
    class GPModel {
    public:
        GPModel()
        {
            init();
        }

        void init()
        {
            _gp_model = GP_t(Params::blackdrops::model_input_dim() + Params::blackdrops::action_dim(), Params::blackdrops::model_pred_dim());
            _limits = Eigen::VectorXd::Ones(Params::blackdrops::model_input_dim());
            _initialized = true;
        }

        void learn(const std::vector<std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd>>& observations, bool only_limits = false)
        {
            std::vector<Eigen::VectorXd> samples, observs;
            Eigen::MatrixXd obs(observations.size(), std::get<2>(observations[0]).size());
            for (size_t i = 0; i < observations.size(); i++) {
                Eigen::VectorXd st, act, pred;
                st = std::get<0>(observations[i]);
                act = std::get<1>(observations[i]);
                pred = std::get<2>(observations[i]);

                Eigen::VectorXd s(st.size() + act.size());
                s.head(st.size()) = st;
                s.tail(act.size()) = act;

                samples.push_back(s);
                obs.row(i) = pred;
                observs.push_back(pred);
                // std::cout << s.transpose() << std::endl;
                // std::cout << pred.transpose() << std::endl;
            }
            _observations = obs;

            Eigen::MatrixXd data = _to_matrix((const std::vector<Eigen::VectorXd>&)samples);
            Eigen::MatrixXd samp = data.block(0, 0, data.rows(), Params::blackdrops::model_input_dim() + Params::blackdrops::action_dim());
            _means = samp.colwise().mean().transpose();
            _sigmas = Eigen::colwise_sig(samp).array().transpose();
            Eigen::VectorXd pl = Eigen::percentile(samp.array().abs(), 5);
            Eigen::VectorXd ph = Eigen::percentile(samp.array().abs(), 95);
            _limits = pl.array().max(ph.array());
            for (int i = 0; i < _limits.size(); i++) {
                if (_limits(i) < 1e-8)
                    _limits(i) = 1.0;
            }

            if (only_limits)
                return;

            Eigen::MatrixXd data2(samples.size(), samples[0].size() + obs.cols());
            for (size_t i = 0; i < samples.size(); i++) {
                data2.block(i, 0, 1, Params::blackdrops::model_input_dim() + Params::blackdrops::action_dim()) = samples[i].transpose(); //.array() / _limits.array();
                data2.block(i, Params::blackdrops::model_input_dim() + Params::blackdrops::action_dim(), 1, Params::blackdrops::model_pred_dim()) = obs.row(i);
            }
            Eigen::write_binary("blackdrops_data.bin", data2);

            std::cout << "GP Samples: " << samples.size() << std::endl;
            if (!_initialized)
                init();

            _gp_model.compute(samples, observs, true);
            _gp_model.optimize_hyperparams();
        }

        void save_data(const std::string& filename) const
        {
            const std::vector<Eigen::VectorXd>& samples = _gp_model.samples();
            Eigen::MatrixXd observations = _observations;

            std::ofstream ofs_data(filename);
            for (size_t i = 0; i < samples.size(); ++i) {
                if (i != 0)
                    ofs_data << std::endl;
                for (int j = 0; j < samples[0].size(); ++j) {
                    ofs_data << samples[i](j) << " ";
                }
                for (int j = 0; j < observations.cols(); ++j) {
                    if (j != 0)
                        ofs_data << " ";
                    ofs_data << observations(i, j);
                }
            }
        }

        std::tuple<Eigen::VectorXd, Eigen::VectorXd> predict(const Eigen::VectorXd& x, bool compute_variance = true) const
        {
            if (compute_variance)
                return _gp_model.query(x);
            return std::make_tuple(_gp_model.mu(x), Eigen::VectorXd::Zero(_gp_model.dim_out()));
        }

        Eigen::MatrixXd samples() const
        {
            return _to_matrix(_gp_model.samples());
        }

        Eigen::MatrixXd observations() const
        {
            return _observations;
        }

        Eigen::VectorXd limits() const
        {
            return _limits;
        }

        std::vector<Eigen::VectorXd> _to_vector(const Eigen::MatrixXd& m) const
        {
            std::vector<Eigen::VectorXd> result(m.rows());
            for (size_t i = 0; i < result.size(); ++i) {
                result[i] = m.row(i);
            }
            return result;
        }
        std::vector<Eigen::VectorXd> _to_vector(Eigen::MatrixXd& m) const { return _to_vector(m); }

        Eigen::MatrixXd _to_matrix(const std::vector<Eigen::VectorXd>& xs) const
        {
            Eigen::MatrixXd result(xs.size(), xs[0].size());
            for (size_t i = 0; i < (size_t)result.rows(); ++i) {
                result.row(i) = xs[i];
            }
            return result;
        }

        Eigen::MatrixXd _to_matrix(std::vector<Eigen::VectorXd>& xs) const { return _to_matrix(xs); }

    protected:
        GP_t _gp_model;
        bool _initialized = false;
        Eigen::MatrixXd _observations;
        Eigen::VectorXd _means, _sigmas, _limits;
    };
} // namespace blackdrops

#endif
