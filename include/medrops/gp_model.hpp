#ifndef MEDROPS_GP_MODEL_HPP
#define MEDROPS_GP_MODEL_HPP

namespace Eigen {
    template<class Matrix>
    void write_binary(const char* filename, const Matrix& matrix) {
        std::ofstream out(filename,std::ios::out | std::ios::binary | std::ios::trunc);
        typename Matrix::Index rows=matrix.rows(), cols=matrix.cols();
        out.write((char*) (&rows), sizeof(typename Matrix::Index));
        out.write((char*) (&cols), sizeof(typename Matrix::Index));
        out.write((char*) matrix.data(), rows*cols*sizeof(typename Matrix::Scalar) );
        out.close();
    }

    template<class Matrix>
    void read_binary(const char* filename, Matrix& matrix){
        std::ifstream in(filename,std::ios::in | std::ios::binary);
        typename Matrix::Index rows=0, cols=0;
        in.read((char*) (&rows),sizeof(typename Matrix::Index));
        in.read((char*) (&cols),sizeof(typename Matrix::Index));
        matrix.resize(rows, cols);
        in.read( (char *) matrix.data() , rows*cols*sizeof(typename Matrix::Scalar) );
        in.close();
    }
}

namespace medrops {

    template <typename Params, typename GP>
    class GPModel {
    public:

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
            _gp_models = std::vector<GP>(std::get<2>(observations[0]).size());
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
                // obs.push_back(pred);
                obs.row(i) = pred;
            }

            // Eigen::VectorXd p_in(7);
            // p_in << 1.03279, 1.1477, 1.23194, 1.00744, 1.17817, 2.71037, 6.48436e-18;
            // _gp_models.kernel_function().set_h_params(p_in);

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
                _gp_models[i].compute(samples, _to_vector(obs.col(i)), noises);
                _gp_models[i].optimize_hyperparams();
            });

            for (size_t i = 0; i < (size_t) obs.cols(); ++i) {
                // Print hparams in logspace
                Eigen::VectorXd p = _gp_models[i].kernel_function().h_params();
                p.segment(0,p.size()-1) = p.segment(0, p.size()-1).array().exp();
                p(p.size()-1) = std::exp(2*p(p.size()-1));
                std::cout << p.array().transpose() << std::endl;
            }

            // Loading test
            // std::cout << std::endl;
            // Eigen::MatrixXd data_comp;
            // Eigen::read_binary("medrops_data.bin", data_comp);
            // std::vector<Eigen::VectorXd> samples_comp(data_comp.rows());
            // Eigen::MatrixXd observations_comp(data_comp.rows(), 4);
            // for (size_t i = 0; i < data_comp.rows(); i++) {
            //     samples_comp[i] = data_comp.row(i).segment(0, 6);
            //     observations_comp.row(i) = data_comp.row(i).segment(6, 4);
            // }

            // std::cout << samples_comp.size() << " " << samples_comp[0].size() << std::endl;
            // for (size_t i = 0; i < (size_t) observations_comp.cols(); ++i) {
            //     _gp_models[i].compute(samples_comp, observations_comp.col(i), noises);
            //     _gp_models[i].optimize_hyperparams();
            //     Eigen::VectorXd p = _gp_models[i].kernel_function().h_params();

            //     // Print hparams in logspace
            //     p.segment(0,p.size()-1) = p.segment(0, p.size()-1).array().exp();
            //     p(p.size()-1) = std::exp(2*p(p.size()-1));
            //     std::cout << p.array().transpose() << std::endl;
            // }
        }

        void save_data() const {
            const std::vector<Eigen::VectorXd>& samples = _gp_models[0].samples();
            Eigen::MatrixXd observations(samples.size(), _gp_models.size());
            for (size_t i = 0; i < _gp_models.size(); ++i) {
                observations.col(i) = _gp_models[i].observations().col(0);
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
                std::tie(m, s) = _gp_models[i].query(x);
                ms(i) = m(0);
                ss(i) = s;
            });
            return std::make_tuple(ms, ss);
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

        std::vector<GP> _gp_models;
    };
}

#endif
