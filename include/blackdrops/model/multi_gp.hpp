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
//| This software is a computer library whose purpose is to optimize continuous,
//| black-box functions. It mainly implements Gaussian processes and Bayesian
//| optimization.
//| Main repository: http://github.com/resibots/blackdrops
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
#ifndef BLACKDROPS_MODEL_MULTI_GP_HPP
#define BLACKDROPS_MODEL_MULTI_GP_HPP

#include <limbo/mean/null_function.hpp>
#include <limbo/model/gp.hpp>

namespace blackdrops {
    namespace model {
        template <typename Params, template <typename, typename, typename, typename> class GPClass, typename KernelFunction, typename MeanFunction, class HyperParamsOptimizer = limbo::model::gp::NoLFOpt<Params>>
        class MultiGP {
        public:
            using GP_t = GPClass<Params, KernelFunction, limbo::mean::NullFunction<Params>, limbo::model::gp::NoLFOpt<Params>>;

            /// useful because the model might be created before knowing anything about the process
            MultiGP() : _dim_in(-1), _dim_out(-1)
            {
            }

            /// useful because the model might be created  before having samples
            MultiGP(int dim_in, int dim_out)
                : _dim_in(dim_in), _dim_out(dim_out), _mean_function(dim_out)
            {
                _gp_models.resize(_dim_out);
                for (int i = 0; i < _dim_out; i++) {
                    _gp_models[i] = GP_t(_dim_in, 1);
                    // _gp_models[i].mean_function().set_id(i);
                }
            }

            /// Compute the GP from samples and observations. This call needs to be explicit!
            void compute(const std::vector<Eigen::VectorXd>& samples,
                const std::vector<Eigen::VectorXd>& observations, bool compute_kernel = true)
            {
                if (_dim_in != samples[0].size()) {
                    _dim_in = samples[0].size();
                }

                if (_dim_out != observations[0].size()) {
                    _dim_out = observations[0].size();
                }

                if ((int)_gp_models.size() != _dim_out) {
                    _gp_models.resize(_dim_out);
                    for (int i = 0; i < _dim_out; i++)
                        _gp_models[i] = GP_t(_dim_in, 1);
                }

                _samples = samples;
                _observations = observations;

                std::vector<std::vector<Eigen::VectorXd>> obs(_dim_out);

                for (size_t j = 0; j < observations.size(); j++) {
                    Eigen::VectorXd mean_vector = _mean_function(samples[j], *this);
                    assert(mean_vector.size() == _dim_out);
                    for (int i = 0; i < _dim_out; i++) {
                        obs[i].push_back(limbo::tools::make_vector(observations[j][i] - mean_vector[i]));
                    }
                }

                tbb::parallel_for(size_t(0), (size_t)_dim_out, size_t(1), [&](size_t i) {
                    // _gp_models[i].mean_function().set_id(i);
                    _gp_models[i].compute(samples, obs[i], compute_kernel);
                });
            }

            void optimize_hyperparams()
            {
                _hp_optimize(*this);
            }

            const MeanFunction& mean_function() const { return _mean_function; }

            MeanFunction& mean_function() { return _mean_function; }

            void add_sample(const Eigen::VectorXd& sample, const Eigen::VectorXd& observation)
            {
                // std::cerr << "[WARNING-ParallelGP]: add_sample might not work properly if the params of MeanFunction are changed between function calls!" << std::endl;
                if (_gp_models.size() == 0) {
                    if (_dim_in != sample.size()) {
                        _dim_in = sample.size();
                    }
                    if (_dim_out != observation.size()) {
                        _dim_out = observation.size();
                        _gp_models.resize(_dim_out);
                        for (int i = 0; i < _dim_out; i++)
                            _gp_models[i] = GP_t(_dim_in, 1);
                    }
                }
                else {
                    assert(sample.size() == _dim_in);
                    assert(observation.size() == _dim_out);
                }

                _samples.push_back(sample);
                _observations.push_back(observation);

                Eigen::VectorXd mean_vector = _mean_function(sample, *this);
                assert(mean_vector.size() == _dim_out);

                tbb::parallel_for(size_t(0), (size_t)_dim_out, size_t(1), [&](size_t i) {
                    // _gp_models[i].mean_function().set_id(i);
                    _gp_models[i].add_sample(sample, limbo::tools::make_vector(observation[i] - mean_vector[i]));
                });
            }

            std::tuple<Eigen::VectorXd, Eigen::VectorXd> query(const Eigen::VectorXd& v) const
            {
                Eigen::VectorXd mu(_dim_out);
                Eigen::VectorXd sigma(_dim_out);

                Eigen::VectorXd mean_vector = _mean_function(v, *this);

#ifndef ONLYMI
                tbb::parallel_for(size_t(0), (size_t)_dim_out, size_t(1), [&](size_t i) {
                    Eigen::VectorXd tmp;
                    std::tie(tmp, sigma(i)) = _gp_models[i].query(v);
                    mu(i) = tmp(0) + mean_vector(i);
                });

                // TO-DO: Fix that
                if (_gp_models[0].samples().size() == 0)
                    sigma = Eigen::VectorXd::Zero(_dim_out);
#else
                mu = mean_vector;
                sigma = Eigen::VectorXd::Zero(_dim_out);
#endif

                return std::make_tuple(mu, sigma);
            }

            Eigen::VectorXd mu(const Eigen::VectorXd& v) const
            {
                Eigen::VectorXd mu(_dim_out);
                Eigen::VectorXd mean_vector = _mean_function(v, *this);

                tbb::parallel_for(size_t(0), (size_t)_dim_out, size_t(1), [&](size_t i) {
                    mu(i) = _gp_models[i].mu(v) + mean_vector(i);
                });

                return mu;
            }

            Eigen::VectorXd sigma(const Eigen::VectorXd& v) const
            {
                Eigen::VectorXd sigma(_dim_out);

                tbb::parallel_for(size_t(0), (size_t)_dim_out, size_t(1), [&](size_t i) {
                    sigma(i) = _gp_models[i].sigma(v);
                });

                return sigma;
            }

            /// return the number of dimensions of the input
            int dim_in() const
            {
                assert(_dim_in != -1); // need to compute first !
                return _dim_in;
            }

            /// return the number of dimensions of the output
            int dim_out() const
            {
                assert(_dim_out != -1); // need to compute first !
                return _dim_out;
            }

            /// return the number of samples used to compute the GP
            int nb_samples() const
            {
                assert(_gp_models.size());
                return _gp_models[0].samples().size();
            }

            ///  recomputes the GP
            void recompute(bool update_obs_mean = true, bool update_full_kernel = true)
            {
                if (update_obs_mean)
                    return compute(_samples, _observations, update_full_kernel);

                tbb::parallel_for(size_t(0), (size_t)_dim_out, size_t(1), [&](size_t i) {
                    _gp_models[i].recompute(update_obs_mean, update_full_kernel);
                });
            }

            /// return the list of samples that have been tested so far
            const std::vector<Eigen::VectorXd>& samples() const
            {
                assert(_gp_models.size());
                return _gp_models[0].samples();
            }

            std::vector<GP_t> gp_models() const
            {
                return _gp_models;
            }

            std::vector<GP_t>& gp_models()
            {
                return _gp_models;
            }

        protected:
            std::vector<GP_t> _gp_models;
            int _dim_in, _dim_out;
            HyperParamsOptimizer _hp_optimize;
            MeanFunction _mean_function;
            std::vector<Eigen::VectorXd> _samples, _observations;
        };
    }
}
#endif
