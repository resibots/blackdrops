#ifndef MEDROPS_EXP_ARD_NOISE_HPP
#define MEDROPS_EXP_ARD_NOISE_HPP

namespace medrops {

    template <typename Params>
    struct SquaredExpARDNoise {
        SquaredExpARDNoise(int dim = 1) : _sf2(1), _noise(0.01), _ell(dim), _input_dim(dim)
        {
            Eigen::VectorXd p = Eigen::VectorXd::Zero(_ell.size() + 2);
            p(_ell.size() + 1) = std::log(0.1);
            this->set_h_params(p);
        }

        size_t h_params_size() const { return _ell.size() + 2; }

        // Return the hyper parameters in log-space
        const Eigen::VectorXd& h_params() const { return _h_params; }

        // We expect the input parameters to be in log-space
        void set_h_params(const Eigen::VectorXd& p)
        {
            _h_params = p;
            for (size_t i = 0; i < _input_dim; ++i)
                _ell(i) = std::exp(p(i));
            _sf2 = std::exp(2 * p(_input_dim));
            _noise = std::exp(2 * p(_input_dim + 1));
        }

        Eigen::VectorXd grad(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2) const
        {
            Eigen::VectorXd grad(_input_dim + 2);
            Eigen::VectorXd z = (x1 - x2).cwiseQuotient(_ell).array().square();
            double k = _sf2 * std::exp(-0.5 * z.sum());
            grad.head(_input_dim) = z * k;
            grad(_input_dim) = 2 * k;
            grad(_input_dim + 1) = 0.0;
            if ((x1 - x2).norm() < 1e-5)
                grad(_input_dim + 1) = 2 * _noise;
            return grad;
        }

        double operator()(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2) const
        {
            assert(x1.size() == _ell.size());
            double z = (x1 - x2).cwiseQuotient(_ell).squaredNorm();
            return _sf2 * std::exp(-0.5 * z);
        }

        const Eigen::VectorXd& ell() const { return _ell; }

    protected:
        double _sf2, _noise;
        Eigen::VectorXd _ell;
        size_t _input_dim;
        Eigen::VectorXd _h_params;
    };
}

#endif
