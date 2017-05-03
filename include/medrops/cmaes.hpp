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
                : libcmaes::BIPOPCMAStrategy<Covariance, TGenoPheno>(func, parameters) {}

            ~customCMAStrategy() {}

            void tell()
            {
                using eostrat = libcmaes::ESOStrategy<libcmaes::CMAParameters<TGenoPheno>, libcmaes::CMASolutions, libcmaes::CMAStopCriteria<TGenoPheno>>;
#ifdef HAVE_DEBUG
                std::chrono::time_point<std::chrono::system_clock> tstart = std::chrono::system_clock::now();
#endif

                // sort candidates.
                if (!eostrat::_parameters.get_uh())
                    eostrat::_solutions.sort_candidates();
                else
                    eostrat::uncertainty_handling();

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
// using eostrat = libcmaes::ESOStrategy<libcmaes::CMAParameters<TGenoPheno>, libcmaes::CMASolutions, libcmaes::CMAStopCriteria<TGenoPheno>>;
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

                // evaluation step of uncertainty handling scheme.
                if (this->_parameters.get_uh()) {
                    // this->perform_uh(candidates, phenocandidates, nfcalls);
                    dMat candidates_uh;
                    this->select_candidates_uh(candidates, phenocandidates, candidates_uh);
                    std::vector<libcmaes::RankedCandidate> nvcandidates;
                    this->eval_candidates_uh(candidates, candidates_uh, nvcandidates, nfcalls);
                    this->set_candidates_uh(nvcandidates);
                }

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

            void eval_candidates_uh(const dMat& candidates, const dMat& candidates_uh, std::vector<libcmaes::RankedCandidate>& nvcandidates, int& nfcalls)
            {
                // creation of #candidates.cols() same RankedCandidates --- should not take a lot of time
                nvcandidates = std::vector<libcmaes::RankedCandidate>(candidates.cols(), libcmaes::RankedCandidate(0.0, this->_solutions.candidates().at(0), 0));
                // re-evaluate
                // for (int r = 0; r < candidates.cols(); r++) {
                tools::par::loop(0, candidates.cols(), [&](int r) {
                    if (r < this->_solutions.lambda_reev()) {
                        double nfvalue = this->_func(candidates_uh.col(r).data(), candidates_uh.rows());
                        nvcandidates[r] = libcmaes::RankedCandidate(nfvalue, this->_solutions.candidates().at(r), r);
                        nfcalls++;
                    }
                    else
                      nvcandidates[r] = libcmaes::RankedCandidate(this->_solutions.candidates().at(r).get_fvalue(), this->_solutions.candidates().at(r), r);
                });
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

                if (bounded)
                    return _opt_bounded(f_cmaes, dim, init);
                else
                    return _opt_unbounded(f_cmaes, dim, init);
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

            // F is a CMA-ES style function, not our function
            template <typename F>
            Eigen::VectorXd _opt_unbounded(F& f_cmaes, int dim, const Eigen::VectorXd& init) const
            {
                using namespace libcmaes;
                // initial step-size, i.e. estimated initial parameter error.
                double sigma = 5.0;
                std::vector<double> x0(init.data(), init.data() + init.size());

                CMAParameters<> cmaparams(x0, sigma);
                _set_common_params(cmaparams, dim);

                // the optimization itself
                CMASolutions cmasols = run_cmaes<>(f_cmaes, cmaparams);
                return cmasols.get_best_seen_candidate().get_x_dvec();
            }

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
                cmaparams.set_uh(Params::opt_cmaes::handle_uncertainty());
                cmaparams.set_quiet(!Params::opt_cmaes::verbose());
            }
        };
    }
}
#endif
#endif
