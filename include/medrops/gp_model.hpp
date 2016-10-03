#ifndef MEDROPS_GP_MODEL_HPP
#define MEDROPS_GP_MODEL_HPP

namespace medrops {

    template <typename Params, typename GP>
    class GPModel {
    public:
        void learn(const std::vector<std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd>>& observations)
        {
            std::vector<Eigen::VectorXd> samples, obs;
            for (size_t i = 0; i < observations.size(); i++) {
                Eigen::VectorXd st, act, pred;
                st = std::get<0>(observations[i]);
                act = std::get<1>(observations[i]);
                pred = std::get<2>(observations[i]);

                Eigen::VectorXd s(st.size() + act.size());
                s.head(st.size()) = st;
                s.tail(act.size()) = act;

                samples.push_back(s);
                obs.push_back(pred);
            }

            _gp_model.compute(samples, obs, Eigen::VectorXd::Constant(samples.size(), Params::gp_model::noise()));
            _gp_model.optimize_hyperparams();
            _gp_model.recompute(false);
        }

        std::tuple<Eigen::VectorXd, double> predict(const Eigen::VectorXd& x) const
        {
            return _gp_model.query(x);
        }

    protected:
        GP _gp_model;
    };
}

#endif
