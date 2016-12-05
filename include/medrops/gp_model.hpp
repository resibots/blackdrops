#ifndef MEDROPS_GP_MODEL_HPP
#define MEDROPS_GP_MODEL_HPP

#include "binary_matrix.hpp"

namespace medrops {

    template <typename Params, typename GP>
    class GPModel {
    public:

        GPModel() {
          _gp_models = std::vector<std::shared_ptr<GP>>(4);
          for (size_t i = 0; i < _gp_models.size(); i++) {
            _gp_models[i] = std::make_shared<GP>(5, 1);
            #ifdef INTACT
              _gp_models[i]->mean_function().set_id(i);
            #endif
          }
        }

        inline double angle_dist(double a, double b)
        {
            double theta = b - a;
            while (theta < -M_PI)
                theta += 2 * M_PI;
            while (theta > M_PI)
                theta -= 2 * M_PI;
            return theta;
        }
        double get_reward(const Eigen::VectorXd& from_state, const Eigen::VectorXd& action, const Eigen::VectorXd& to_state) const
        {
            double s_c_sq = 0.25 * 0.25;
            double dx = angle_dist(to_state(3), Params::goal_pos());
            // double dy = to_state(2) - Params::goal_vel();
            // double dz = to_state(1) - Params::goal_vel_x();
            double dw = to_state(0) - Params::goal_pos_x();

            return std::exp(-0.5 / s_c_sq * (dx * dx /*+ dy * dy + dz * dz*/ + dw * dw));
        }

        void learn(const std::vector<std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd>>& observations)
        {
            #ifdef INTACT
              if (_initialized) return;
            #endif

            #ifndef DATA

              std::vector<Eigen::VectorXd> samples;
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
              }

              std::ofstream ofs_data("medrops_gp_data.dat");
              ofs_data << std::setprecision(std::numeric_limits<long double>::digits10 + 1);
              for (size_t i = 0; i < samples.size(); ++i) {
                  if (i != 0) ofs_data << std::endl;
                  for (size_t j = 0; j < samples[0].size(); ++j) {
                      ofs_data << samples[i](j) << " ";
                  }
                  for (size_t j = 0; j < obs.cols(); ++j) {
                      if (j != 0) ofs_data << " ";
                      ofs_data << obs(i,j);
                  }
              }

              Eigen::MatrixXd data(samples.size(), samples[0].size()+obs.cols());
              for (size_t i = 0; i < samples.size(); i++) {
                  data.block(i, 0, 1, 6) = samples[i].transpose();
                  data.block(i, 6, 1, 4) = obs.row(i);
              }
              Eigen::write_binary("medrops_data.bin", data);

              std::cout << "GP Samples: " << samples.size() << std::endl;
              Eigen::VectorXd noises = Eigen::VectorXd::Constant(samples.size(), Params::gp_model::noise());
              tbb::parallel_for(size_t(0), (size_t)obs.cols(), size_t(1), [&](size_t i) {
                  _gp_models[i]->compute(samples, _to_vector(obs.col(i)), noises);
                  _gp_models[i]->optimize_hyperparams();
              });

              for (size_t i = 0; i < (size_t) obs.cols(); ++i) {
                  // Print hparams in logspace
                  Eigen::VectorXd p = _gp_models[i]->kernel_function().h_params();
                  p.segment(0,p.size()-1) = p.segment(0, p.size()-1).array().exp();
                  p(p.size()-1) = std::exp(2*p(p.size()-1));
                  std::cout << p.array().transpose() << std::endl;
              }

            #else

              // Loading test
              std::cout << std::endl;
              Eigen::MatrixXd data_comp;
              Eigen::read_binary("medrops_data_01_12_2016.bin", data_comp);

              size_t limit = 120;
              std::cout << "Loading " << limit << "/" << data_comp.rows() << " rows from file." << std::endl;

              std::vector<Eigen::VectorXd> samples_comp(limit);
              Eigen::MatrixXd observations_comp(limit, 4);
              for (size_t i = 0; i < limit; i++) {
                  samples_comp[i] = data_comp.row(i).segment(0, 6);
                  observations_comp.row(i) = data_comp.row(i).segment(6, 4);
              }

              Eigen::VectorXd noises = Eigen::VectorXd::Constant(samples_comp.size(), Params::gp_model::noise());
              tbb::parallel_for(size_t(0), (size_t) observations_comp.cols(), size_t(1), [&](size_t i) {
                  _gp_models[i]->compute(samples_comp, _to_vector(observations_comp.col(i)), noises);
                  _gp_models[i]->optimize_hyperparams();
                  std::cout << "Computation for gp " << i << " ended." << std::endl;

                  // Print hparams in logspace
                  // Eigen::VectorXd p = _gp_models[i]->kernel_function().h_params();
                  // p.segment(0,p.size()-1) = p.segment(0, p.size()-1).array().exp();
                  // p(p.size()-1) = std::exp(2*p(p.size()-1));
                  // std::cout << p.array().transpose() << std::endl;
              });

            #endif
            _initialized = true;
        }

        void save_data() const {
            const std::vector<Eigen::VectorXd>& samples = _gp_models[0].samples();
            Eigen::MatrixXd observations(samples.size(), _gp_models.size());
            for (size_t i = 0; i < _gp_models.size(); ++i) {
                observations.col(i) = _gp_models[i]->observations().col(0);
            }

            std::ofstream ofs_data("medrops_gp_data.dat");
            for (size_t i = 0; i < samples.size(); ++i) {
                if (i != 0) ofs_data << std::endl;
                for (size_t j = 0; j < samples[0].size(); ++j) {
                    ofs_data << samples[i](j) << " ";
                }
                for (size_t j = 0; j < observations.cols(); ++j) {
                    if (j != 0) ofs_data << " ";
                    ofs_data << observations(i,j);
                }
            }
        }

        std::tuple<Eigen::VectorXd, double> predict(const Eigen::VectorXd& x) const
        {
            Eigen::VectorXd ms;
            Eigen::VectorXd ss;
            std::tie(ms, ss) = predictm(x);
            return std::make_tuple(ms, ss.mean());
        }

        std::tuple<Eigen::VectorXd, Eigen::VectorXd> predictm(const Eigen::VectorXd& x) const
        {
            Eigen::VectorXd ms(_gp_models.size());
            Eigen::VectorXd ss(_gp_models.size());
            tbb::parallel_for(size_t(0), _gp_models.size(), size_t(1), [&](size_t i) {
                double s;
                Eigen::VectorXd m;
                std::tie(m, s) = _gp_models[i]->query(x);
                ms(i) = m(0);
                ss(i) = s;
            });
            return std::make_tuple(ms, ss);
        }

        Eigen::MatrixXd samples() const {
          return _to_matrix(_gp_models[0]->samples());
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

        std::vector<std::shared_ptr<GP>> _gp_models;
        bool _initialized = false;
    };
}

#endif
