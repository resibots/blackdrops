#ifndef MEDROPS_CMAES_HPP
#define MEDROPS_CMAES_HPP

#include <Eigen/Core>
#include <iostream>
#include <vector>

#include <limbo/opt/optimizer.hpp>
#include <limbo/tools/macros.hpp>
#include <limbo/tools/parallel.hpp>

#ifndef USE_LIBCMAES
#warning NO libcmaes support
#else

#include <libcmaes/cmaes.h>

namespace limbo {
    namespace opt {

        template <typename Params, typename Covariance, typename TGenoPheno>
        class customCMAStrategy : public libcmaes::BIPOPCMAStrategy<Covariance, TGenoPheno> {
        public:
            customCMAStrategy(libcmaes::FitFunc& func,
                libcmaes::CMAParameters<TGenoPheno>& parameters)
                : libcmaes::BIPOPCMAStrategy<Covariance, TGenoPheno>(func, parameters)
            {
                _t_limit = 3;
                _delta = 0.1;
                _alpha = 1.5;
            }

            ~customCMAStrategy() {}

            void tell()
            {
                using eostrat = libcmaes::ESOStrategy<libcmaes::CMAParameters<TGenoPheno>, libcmaes::CMASolutions, libcmaes::CMAStopCriteria<TGenoPheno>>;
#ifdef HAVE_DEBUG
                std::chrono::time_point<std::chrono::system_clock> tstart = std::chrono::system_clock::now();
#endif

                // sort candidates.
                // if (!eostrat::_parameters.get_uh())
                eostrat::_solutions.sort_candidates();
                // else
                //     eostrat::uncertainty_handling();

                // call on tpa computation of s(t)
                if (eostrat::_parameters.get_tpa() == 2 && eostrat::_niter > 0)
                    eostrat::tpa_update();

                // update function value history, as needed.
                eostrat::_solutions.update_best_candidates();

                // CMA-ES update, depends on the selected 'flavor'.
                Covariance::update(eostrat::_parameters, this->_esolver, eostrat::_solutions);

                if (eostrat::_parameters.get_uh())
                    if (eostrat::_solutions.suh() > 0.0)
                        eostrat::_solutions.set_sigma(eostrat::_solutions.sigma() * eostrat::_parameters.alphathuh());

                // other stuff.
                if (!eostrat::_parameters.is_sep() && !eostrat::_parameters.is_vd())
                    eostrat::_solutions.update_eigenv(this->_esolver._eigenSolver.eigenvalues(),
                        this->_esolver._eigenSolver.eigenvectors());
                else
                    eostrat::_solutions.update_eigenv(eostrat::_solutions.sepcov(),
                        dMat::Constant(eostrat::_parameters.dim(), 1, 1.0));
#ifdef HAVE_DEBUG
                std::chrono::time_point<std::chrono::system_clock> tstop = std::chrono::system_clock::now();
                eostrat::_solutions._elapsed_tell = std::chrono::duration_cast<std::chrono::milliseconds>(tstop - tstart).count();
#endif
            }

            void eval(const dMat& candidates,
                const dMat& phenocandidates = dMat(0, 0))
            {
#ifdef HAVE_DEBUG
                std::chrono::time_point<std::chrono::system_clock> tstart = std::chrono::system_clock::now();
#endif
                // one candidate per row.
                // #pragma omp parallel for if (_parameters._mt_feval)
                // for (int r = 0; r < candidates.cols(); r++)
                tools::par::loop(0, candidates.cols(), [&](size_t r) {
                    this->_solutions.candidates().at(r).set_x(candidates.col(r));
                    this->_solutions.candidates().at(r).set_id(r);
                    if (phenocandidates.size())
                        this->_solutions.candidates().at(r).set_fvalue(this->_func(phenocandidates.col(r).data(), candidates.rows()));
                    else
                        this->_solutions.candidates().at(r).set_fvalue(this->_func(candidates.col(r).data(), candidates.rows()));

                    //std::cerr << "candidate x: " << _solutions._candidates.at(r)._x.transpose() << std::endl;
                });
                int nfcalls = candidates.cols();

                if (Params::opt_cmaes::handle_uncertainty()) {
                    // new uncertainty
                    int t_max = 20;
                    // _t_limit = 3;
                    dMat selected, discarded, undecided;
                    std::map<int, int> ids;
                    std::vector<double> mean, lower, upper;
                    for (int r = 0; r < candidates.cols(); r++) {
                        mean.push_back(this->_solutions.candidates().at(r).get_fvalue());
                        ids[r] = r;
                        lower.push_back(Params::opt_cmaes::a());
                        upper.push_back(Params::opt_cmaes::b());
                    }

                    undecided = candidates;

                    std::vector<int> u;
                    u.push_back(undecided.cols());
                    double R = std::abs(Params::opt_cmaes::a() - Params::opt_cmaes::b());
                    int t = 1;
                    std::vector<int> calls(undecided.cols(), 1);
                    while (t < _t_limit && selected.cols() < this->_parameters.mu()) {
                        t = t + 1;
                        // std::cout << t << ": " << selected.cols() << " " << undecided.cols() << " " << discarded.cols() << " " << this->_parameters.lambda() << " " << this->_parameters.mu() << std::endl;
                        // std::cin.get();
                        u.push_back(undecided.cols());

                        int n_b = 0;
                        for (int k = 0; k < t; k++) {
                            n_b += u[k] + (_t_limit - t + 1) * u[t - 1];
                        }

                        // for (int r = 0; r < undecided.cols(); r++) {
                        tools::par::loop(0, undecided.cols(), [&](size_t r) {
                        double val;
                        if (phenocandidates.size())
                            val = this->_func(phenocandidates.col(ids[r]).data(), candidates.rows());
                        else
                            val = this->_func(candidates.col(ids[r]).data(), candidates.rows());
                        calls[ids[r]]++;
                        mean[ids[r]] = (mean[ids[r]] + val) / 2.0;
                        this->_solutions.candidates().at(ids[r]).set_fvalue(mean[ids[r]]);
                        double c = R * std::sqrt((std::log(2 * n_b) - std::log(_delta)) / (2.0 * t));
                        lower[ids[r]] = std::max(lower[ids[r]], mean[ids[r]] - c);
                        upper[ids[r]] = std::min(upper[ids[r]], mean[ids[r]] + c);
                        });

                        nfcalls += undecided.cols();

                        for (int r = 0; r < undecided.cols(); r++) {
                            int c1 = 0, c2 = 0;
                            for (int j = 0; j < undecided.cols(); j++) {
                                // TO-DO: Check minimize/maximize
                                if (lower[ids[r]] < upper[ids[j]])
                                    c1++;
                                if (upper[ids[r]] > lower[ids[j]])
                                    c2++;
                            }

                            if (c1 >= (this->_parameters.lambda() - this->_parameters.mu() - discarded.cols())) {
                                Eigen::VectorXd col = undecided.col(r);
                                removeColumn(undecided, r);
                                // update ids
                                for (int k = r; k < undecided.cols() + 1; k++) {
                                    ids[k] = ids[k + 1];
                                }

                                // add to selected
                                selected.conservativeResize(undecided.rows(), selected.cols() + 1);
                                selected.col(selected.cols() - 1) = col;
                                r--;
                            }

                            else if (c2 >= (this->_parameters.mu() - selected.cols())) {
                                Eigen::VectorXd col = undecided.col(r);
                                removeColumn(undecided, r);
                                // update ids
                                for (int k = r; k < undecided.cols() + 1; k++) {
                                    ids[k] = ids[k + 1];
                                }

                                // add to discarded
                                discarded.conservativeResize(undecided.rows(), discarded.cols() + 1);
                                discarded.col(discarded.cols() - 1) = col;
                                r--;
                            }
                        }
                    }

                    // std::cout << "t_limit: " << _t_limit << std::endl;
                    // for (int r = 0; r < calls.size(); r++) {
                    //     std::cout << calls[r] << std::endl;
                    // }
                    // std::cout << "-------------------------" << std::endl;
                    // for (int r = 0; r < mean.size(); r++) {
                    //     std::cout << mean[r] << std::endl;
                    // }
                    // std::cin.get();

                    if (selected.cols() == this->_parameters.mu())
                        _t_limit = std::max(3, int(1.0 / (_alpha * _t_limit)));
                    else
                        _t_limit = std::min(int(_alpha * _t_limit), t_max);
                }

                // std::cout << _t_limit << std::endl;

                // // evaluation step of uncertainty handling scheme.
                // if (this->_parameters.get_uh()) {
                //     this->perform_uh(candidates, phenocandidates, nfcalls);
                // }

                // if an elitist is active, reinject initial solution as needed.
                if (this->_niter > 0 && (this->_parameters.elitist() || this->_parameters.initial_elitist() || (this->_initial_elitist && this->_parameters.initial_elitist_on_restart()))) {
                    // get reference values.
                    double ref_fvalue = std::numeric_limits<double>::max();
                    libcmaes::Candidate ref_candidate;

                    if (this->_parameters.initial_elitist_on_restart() || this->_parameters.initial_elitist()) {
                        ref_fvalue = this->_solutions.initial_candidate().get_fvalue();
                        ref_candidate = this->_solutions.initial_candidate();
                    }
                    else if (this->_parameters.elitist()) {
                        ref_fvalue = this->_solutions.get_best_seen_candidate().get_fvalue();
                        ref_candidate = this->_solutions.get_best_seen_candidate();
                    }

                    // reinject intial solution if half or more points have value above that of the initial point candidate.
                    int count = 0;
                    for (int r = 0; r < candidates.cols(); r++)
                        if (this->_solutions.candidates().at(r).get_fvalue() < ref_fvalue)
                            ++count;
                    if (count / 2.0 < candidates.cols() / 2) {
#ifdef HAVE_DEBUG
                        std::cout << "reinjecting solution=" << ref_fvalue << std::endl;
#endif
                        this->_solutions.candidates().at(1) = ref_candidate;
                    }
                }

                this->update_fevals(nfcalls);

#ifdef HAVE_DEBUG
                std::chrono::time_point<std::chrono::system_clock> tstop = std::chrono::system_clock::now();
                this->_solutions._elapsed_eval = std::chrono::duration_cast<std::chrono::milliseconds>(tstop - tstart).count();
#endif
            }

            void removeRow(Eigen::MatrixXd& matrix, unsigned int rowToRemove)
            {
                unsigned int numRows = matrix.rows() - 1;
                unsigned int numCols = matrix.cols();

                if (rowToRemove < numRows)
                    matrix.block(rowToRemove, 0, numRows - rowToRemove, numCols) = matrix.block(rowToRemove + 1, 0, numRows - rowToRemove, numCols);

                matrix.conservativeResize(numRows, numCols);
            }

            void removeColumn(Eigen::MatrixXd& matrix, unsigned int colToRemove)
            {
                unsigned int numRows = matrix.rows();
                unsigned int numCols = matrix.cols() - 1;

                if (colToRemove < numCols)
                    matrix.block(0, colToRemove, numRows, numCols - colToRemove) = matrix.block(0, colToRemove + 1, numRows, numCols - colToRemove);

                matrix.conservativeResize(numRows, numCols);
            }

        protected:
            int _t_limit;
            double _delta, _alpha;
        };

        /// @ingroup opt
        /// Covariance Matrix Adaptation Evolution Strategy by Hansen et al.
        /// (See: https://www.lri.fr/~hansen/cmaesintro.html)
        /// - our implementation is based on libcmaes (https://github.com/beniz/libcmaes)
        /// - Support bounded and unbounded optimization
        /// - Only available if libcmaes is installed (see the compilation instructions)
        ///
        /// - Parameters :
        ///   - int variant
        ///   - int elitism
        ///   - int restarts
        ///   - double max_fun_evals
        ///   - double fun_tolerance
        ///   - double fun_target
        ///   - bool fun_compute_initial
        ///   - bool handle_uncertainty
        ///   - bool verbose
        ///   - double lb (lower bounds)
        ///   - double ub (upper bounds)
        template <typename Params>
        struct CustomCmaes {
        public:
            template <typename F>
            Eigen::VectorXd operator()(const F& f, const Eigen::VectorXd& init, double bounded) const
            {
                size_t dim = init.size();

                // wrap the function
                libcmaes::FitFunc f_cmaes = [&](const double* x, const int n) {
                Eigen::Map<const Eigen::VectorXd> m(x, n);
                // remember that our optimizers maximize
                return -eval(f, m);
                };

                assert(bounded);

                // if (bounded)
                return _opt_bounded(f_cmaes, dim, init);
                // else
                //     return _opt_unbounded(f_cmaes, dim, init);
            }

        private:
            template <typename TGenoPheno>
            libcmaes::CMASolutions run_cmaes(libcmaes::FitFunc& func,
                libcmaes::CMAParameters<TGenoPheno>& parameters,
                libcmaes::ProgressFunc<libcmaes::CMAParameters<TGenoPheno>, libcmaes::CMASolutions>& pfunc = libcmaes::CMAStrategy<libcmaes::CovarianceUpdate, TGenoPheno>::_defaultPFunc,
                libcmaes::GradFunc gfunc = nullptr,
                const libcmaes::CMASolutions& solutions = libcmaes::CMASolutions(),
                libcmaes::PlotFunc<libcmaes::CMAParameters<TGenoPheno>, libcmaes::CMASolutions>& pffunc = libcmaes::CMAStrategy<libcmaes::CovarianceUpdate, TGenoPheno>::_defaultFPFunc) const
            {
                using namespace libcmaes;
                ESOptimizer<customCMAStrategy<Params, ACovarianceUpdate, TGenoPheno>, CMAParameters<TGenoPheno>, CMASolutions> abipop(func, parameters);
                if (gfunc != nullptr)
                    abipop.set_gradient_func(gfunc);
                abipop.set_progress_func(pfunc);
                abipop.set_plot_func(pffunc);
                abipop.optimize();
                return abipop.get_solutions();
            }

            // // F is a CMA-ES style function, not our function
            // template <typename F>
            // Eigen::VectorXd _opt_unbounded(F& f_cmaes, int dim, const Eigen::VectorXd& init) const
            // {
            //     using namespace libcmaes;
            //     // initial step-size, i.e. estimated initial parameter error.
            //     double sigma = 0.5;
            //     std::vector<double> x0(init.data(), init.data() + init.size());
            //
            //     CMAParameters<> cmaparams(x0, sigma);
            //     _set_common_params(cmaparams, dim);
            //
            //     // the optimization itself
            //     CMASolutions cmasols = run_cmaes<>(f_cmaes, cmaparams);
            //     return cmasols.get_best_seen_candidate().get_x_dvec();
            // }

            // F is a CMA-ES style function, not our function
            template <typename F>
            Eigen::VectorXd _opt_bounded(F& f_cmaes, int dim, const Eigen::VectorXd& init) const
            {
                using namespace libcmaes;
                // create the parameter object
                // boundary_transformation
                double lbounds[dim], ubounds[dim]; // arrays for lower and upper parameter bounds, respectively
                for (int i = 0; i < dim; i++) {
                    lbounds[i] = Params::opt_cmaes::lbound();
                    ubounds[i] = Params::opt_cmaes::ubound();
                }
                GenoPheno<pwqBoundStrategy> gp(lbounds, ubounds, dim);
                // initial step-size, i.e. estimated initial parameter error.
                double sigma = 0.5 * std::abs(Params::opt_cmaes::ubound() - Params::opt_cmaes::lbound());
                std::vector<double> x0(init.data(), init.data() + init.size());
                // -1 for automatically decided lambda, 0 is for random seeding of the internal generator.
                CMAParameters<GenoPheno<pwqBoundStrategy>> cmaparams(dim, &x0.front(), sigma, -1, 0, gp);
                _set_common_params(cmaparams, dim);

                // the optimization itself
                CMASolutions cmasols = run_cmaes<GenoPheno<pwqBoundStrategy>>(f_cmaes, cmaparams);
                //cmasols.print(std::cout, 1, gp);
                //to_f_representation
                return gp.pheno(cmasols.get_best_seen_candidate().get_x_dvec());
            }

            template <typename P>
            void _set_common_params(P& cmaparams, int dim) const
            {
                using namespace libcmaes;

                // set multi-threading to true
                cmaparams.set_mt_feval(true);
                cmaparams.set_algo(Params::opt_cmaes::variant());
                cmaparams.set_restarts(Params::opt_cmaes::restarts());
                cmaparams.set_elitism(Params::opt_cmaes::elitism());

                // if no max fun evals provided, we compute a recommended value
                size_t max_evals = Params::opt_cmaes::max_fun_evals() < 0
                    ? (900.0 * (dim + 3.0) * (dim + 3.0))
                    : Params::opt_cmaes::max_fun_evals();
                cmaparams.set_max_fevals(max_evals);
                // max iteration is here only for security
                cmaparams.set_max_iter(100000);

                if (Params::opt_cmaes::fun_tolerance() == -1) {
                    // we do not know if what is the actual maximum / minimum of the function
                    // therefore we deactivate this stopping criterion
                    cmaparams.set_stopping_criteria(FTARGET, false);
                }
                else {
                    // the FTARGET criteria also allows us to enable ftolerance
                    cmaparams.set_stopping_criteria(FTARGET, true);
                    cmaparams.set_ftolerance(Params::opt_cmaes::fun_tolerance());
                }

                // we allow to set the ftarget parameter
                if (Params::opt_cmaes::fun_target() != -1) {
                    cmaparams.set_stopping_criteria(FTARGET, true);
                    cmaparams.set_ftarget(-Params::opt_cmaes::fun_target());
                }

                // enable stopping criteria by several equalfunvals and maxfevals
                cmaparams.set_stopping_criteria(EQUALFUNVALS, true);
                cmaparams.set_stopping_criteria(MAXFEVALS, true);

                // enable additional criterias to stop
                cmaparams.set_stopping_criteria(TOLX, true);
                cmaparams.set_stopping_criteria(CONDITIONCOV, true);

                // enable or disable different parameters
                cmaparams.set_initial_fvalue(Params::opt_cmaes::fun_compute_initial());
                // cmaparams.set_uh(Params::opt_cmaes::handle_uncertainty());
                cmaparams.set_quiet(!Params::opt_cmaes::verbose());
            }
        };
    }
}
#endif
#endif
