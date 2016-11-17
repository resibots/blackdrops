#ifndef MEDROPS_GP_MODEL_HPP
#define MEDROPS_GP_MODEL_HPP

namespace medrops {

    template <typename Params, typename GP>
    class GPModel {
    public:
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

            std::cout << "GP Samples: " << samples.size() << std::endl;

            // Eigen::VectorXd p_in(7);
            // p_in << 1.03279, 1.1477, 1.23194, 1.00744, 1.17817, 2.71037, 6.48436e-18;
            // _gp_models.kernel_function().set_h_params(p_in);

            for (int i = 0; i < obs.cols(); i++) {
                _gp_models[i].compute(samples, _to_vector(obs.col(i)), Eigen::VectorXd::Constant(samples.size(), Params::gp_model::noise()));
                _gp_models[i].optimize_hyperparams();
                Eigen::VectorXd p = _gp_models[i].kernel_function().h_params();

                // Print hparams in logspace
                p.segment(0,p.size()-1) = p.segment(0, p.size()-1).array().exp();
                p(p.size()-1) = std::exp(2*p(p.size()-1));
                std::cout << p.array().transpose() << std::endl;
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
            for (int i = 0; i < _gp_models.size(); i++) {
                double s;
                Eigen::VectorXd m;
                std::tie(m, s) = _gp_models[i].query(x);
                ms(i) = m(0);
                ss(i) = s;
            }
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
