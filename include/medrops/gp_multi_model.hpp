#ifndef MEDROPS_GPMM_MODEL_HPP
#define MEDROPS_GPMM_MODEL_HPP

namespace limbo {

  namespace defaults {
      struct model_gpmm {
        BO_PARAM(int, threshold, 300);
      };
  }

  namespace model {

    template <typename Params, typename MeanFunction, typename GPLow, typename GPHigh>
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
        _mean_function = MeanFunction(_dim_out);
        _samples_size = 0;
      }

      const MeanFunction& mean_function() const { return _mean_function; }
      MeanFunction& mean_function() { return _mean_function; }

      /// Compute the GP from samples, observation, noise. This call needs to be explicit!
      void compute(const std::vector<Eigen::VectorXd>& samples,
          const std::vector<Eigen::VectorXd>& observations,
          const Eigen::VectorXd& noises)
      {
        if (_dim_out == -1) {
          _dim_in = samples[0].size();
          _dim_out = observations[0].size();
          _mean_function = MeanFunction(_dim_out);
        }
        _samples = samples;
        _samples_size = samples.size();
        if (_samples_size < Params::model_gpmm::threshold()) {
          std::cout << "GP LOW" << std::endl;
          _gp_low->mean_function() = _mean_function;
          _gp_low->compute(samples, observations, noises);
        } else {
          std::cout << "GP HIGH" << std::endl;
          _gp_high->mean_function() = _mean_function;
          _gp_high->compute(samples, observations, noises);
        }
      }

      std::tuple<Eigen::VectorXd, double> query(const Eigen::VectorXd& v) const
      {
        if (_samples_size < Params::model_gpmm::threshold()) {
          return _gp_low->query(v);
        } else {
          return _gp_high->query(v);
        }
      }

      Eigen::VectorXd mu(const Eigen::VectorXd& v) const
      {
        if (_samples_size < Params::model_gpmm::threshold()) {
          return _gp_low->mu(v);
        } else {
          return _gp_high->mu(v);
        }
      }

      double sigma(const Eigen::VectorXd& v) const
      {
        if (_samples_size < Params::model_gpmm::threshold()) {
          return _gp_low->sigma(v);
        } else {
          return _gp_high->sigma(v);
        }
      }

      /// Do not forget to call this if you use hyper-prameters optimization!!
      void optimize_hyperparams()
      {
        if (_samples_size < Params::model_gpmm::threshold()) {
          _gp_low->optimize_hyperparams();
        }
      }

      /// return the list of samples that have been tested so far
      const std::vector<Eigen::VectorXd>& samples() const {
        return _samples;
      }

    private:
      int _dim_in = -1;
      int _dim_out = -1;
      MeanFunction _mean_function;
      std::vector<Eigen::VectorXd> _samples;
      std::shared_ptr<GPLow> _gp_low;
      std::shared_ptr<GPHigh> _gp_high;
      size_t _samples_size;
    };
  }
}

#endif
