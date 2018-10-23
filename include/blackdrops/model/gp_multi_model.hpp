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
#ifndef BLACKDROPS_MODEL_GP_MULTI_MODEL_HPP
#define BLACKDROPS_MODEL_GP_MULTI_MODEL_HPP

#include <Eigen/Core>

namespace blackdrops {
    namespace defaults {
        struct model_gpmm {
            BO_PARAM(int, threshold, 200);
        };
    } // namespace defaults

    namespace model {
        template <typename Params, typename GPLow, typename GPHigh>
        class GPMultiModel {
        public:
            /// useful because the model might be created before knowing anything about the process
            GPMultiModel() : _dim_in(-1), _dim_out(-1)
            {
                _gp_low = std::make_shared<GPLow>();
                _gp_high = std::make_shared<GPHigh>();
                _samples_size = 0;
            }

            /// useful because the model might be created before having samples
            GPMultiModel(int dim_in, int dim_out) : _dim_in(dim_in), _dim_out(dim_out)
            {
                _gp_low = std::make_shared<GPLow>(_dim_in, _dim_out);
                _gp_high = std::make_shared<GPHigh>(_dim_in, _dim_out);
                _samples_size = 0;
            }

            /// Compute the GP from samples, observation, noise. This call needs to be explicit!
            void compute(const std::vector<Eigen::VectorXd>& samples,
                const std::vector<Eigen::VectorXd>& observations, bool compute_kernel = false)
            {
                _samples = samples;
                _samples_size = samples.size();
                if (_samples_size < Params::model_gpmm::threshold()) {
                    std::cout << "GP LOW" << std::endl;
                    _gp_low->compute(samples, observations, compute_kernel);
                }
                else {
                    std::cout << "GP HIGH" << std::endl;
                    _gp_high->compute(samples, observations, compute_kernel);
                }
            }

            std::tuple<Eigen::VectorXd, Eigen::VectorXd> query(const Eigen::VectorXd& v) const
            {
                if (_samples_size < Params::model_gpmm::threshold()) {
                    return _gp_low->query(v);
                }
                else {
                    return _gp_high->query(v);
                }
            }

            Eigen::VectorXd mu(const Eigen::VectorXd& v) const
            {
                if (_samples_size < Params::model_gpmm::threshold()) {
                    return _gp_low->mu(v);
                }
                else {
                    return _gp_high->mu(v);
                }
            }

            Eigen::VectorXd sigma(const Eigen::VectorXd& v) const
            {
                if (_samples_size < Params::model_gpmm::threshold()) {
                    return _gp_low->sigma(v);
                }
                else {
                    return _gp_high->sigma(v);
                }
            }

            /// Do not forget to call this if you use hyper-prameters optimization!!
            void optimize_hyperparams()
            {
                if (_samples_size < Params::model_gpmm::threshold()) {
                    _gp_low->optimize_hyperparams();
                }
                else {
                    _gp_high->optimize_hyperparams();
                }
            }

            /// return the list of samples that have been tested so far
            const std::vector<Eigen::VectorXd>& samples() const
            {
                return _samples;
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

            /// save the parameters and the data for the GP to the archive (text or binary)
            template <typename A>
            void save(const std::string& directory) const
            {
                A archive(directory);
                save(archive);
            }

            /// save the parameters and the data for the GP to the archive (text or binary)
            template <typename A>
            void save(const A& archive) const
            {
                _gp_low->template save<A>(archive.directory() + "/low");
                _gp_high->template save<A>(archive.directory() + "/high");
            }

            /// load the parameters and the data for the GP from the archive (text or binary)
            /// if recompute is true, we do not read the kernel matrix
            /// but we recompute it given the data and the hyperparameters
            template <typename A>
            void load(const std::string& directory, bool recompute = true)
            {
                A archive(directory);
                load(archive, recompute);
            }

            /// load the parameters and the data for the GP from the archive (text or binary)
            /// if recompute is true, we do not read the kernel matrix
            /// but we recompute it given the data and the hyperparameters
            template <typename A>
            void load(const A& archive, bool recompute = true)
            {
                _gp_low->template load<A>(archive.directory() + "/low", recompute);
                _gp_high->template load<A>(archive.directory() + "/high", recompute);
            }

        private:
            int _dim_in = -1;
            int _dim_out = -1;
            std::vector<Eigen::VectorXd> _samples;
            std::shared_ptr<GPLow> _gp_low;
            std::shared_ptr<GPHigh> _gp_high;
            size_t _samples_size;
        };
    } // namespace model
} // namespace blackdrops

#endif
