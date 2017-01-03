#ifndef MEDROPS_GP_POLICY_HPP
#define MEDROPS_GP_POLICY_HPP
#include <limbo/tools/random_generator.hpp>
#include <Eigen/Core>
#include <limbo/tools/macros.hpp>
#include <limbo/limbo.hpp>
#include <limbo/tools.hpp>

///**************** Load csv file to Eigen::MatrixXd ***************************
#include <Eigen/Dense>
#include <vector>
#include <fstream>
using namespace Eigen;
template<typename M>
M load_csv (const std::string & path, const char delim =',') {
    std::ifstream indata;
    indata.open(path);
    if(indata == NULL)
        std::cout<<"Null stream \n";
    std::string line;
    std::vector<double> values;
    uint rows = 0;
    while (std::getline(indata, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        while (std::getline(lineStream, cell, delim)) {
            values.push_back(std::stod(cell));
        }
        ++rows;
    }
    return Map<const Matrix<typename M::Scalar, M::RowsAtCompileTime, M::ColsAtCompileTime, RowMajor>>(values.data(), rows, values.size()/rows);
}

Eigen::VectorXd push_back(Eigen::VectorXd v1, Eigen::VectorXd v2)
{
    Eigen::VectorXd temp(v1.size()+v2.size());

    for(int i=0; i<v1.size(); i++)
    {
        temp(i) = v1(i);

    }
    for(int i=0; i<v2.size(); i++)
    {
        temp(v1.size()+i) = v2(i);
    }
    return temp;
}

Eigen::VectorXd matx2vecx(const Eigen::MatrixXd mat)
{
    Eigen::VectorXd vec(mat.rows()*mat.cols());
    int r,c;
    r = mat.rows();
    c = mat.cols();
    for(int i=0; i<r; i++){
        for(int j=0; j<c; j++){
            vec(i*c+j) = mat(i,j);
        }
    }
    return vec;
}

//************************* end ************************************************

namespace medrops {
    namespace defaults{
        struct gp_policy_defaults {
            BO_PARAM(double, max_u, 10.0); //max action
            BO_PARAM(double, pseudo_samples, 10);
            BO_PARAM(double, noise, 1e-5);
        };
    }
    template <typename Params, typename Model>
    struct GPPolicy {
        using kernel_t = limbo::kernel::SquaredExpARD<Params>;
        //using mean_t = limbo::mean::Data<Params>;
        using mean_t = limbo::mean::Constant<Params>;
        size_t sdim = Params::gp_policy::state_dim(); //input dimension
        size_t ps = Params::gp_policy::pseudo_samples(); //total observations
        using gp_t = limbo::model::GP<Params, kernel_t, mean_t>;

        GPPolicy()
        {
            _random = false;
             size_t sdim = Params::nn_policy::state_dim();
             size_t ps = Params::gp_policy::pseudo_samples();
             _params = Eigen::VectorXd::Zero((sdim+1)*ps+sdim);
             this -> load_parameters_from_file();
        }

        void load_parameters_from_file()
        {
            MatrixXd A = load_csv<MatrixXd>("./exp/medrops/data/pseudoInputs.csv", '\t');
            Eigen::VectorXd ps = matx2vecx(A);
            MatrixXd B = load_csv<MatrixXd>("./exp/medrops/data/pseudoTargets.csv", '\t');
            Eigen::VectorXd pt = matx2vecx(B);
            MatrixXd C = load_csv<MatrixXd>("./exp/medrops/data/hyperParams.csv", '\t');
            Eigen::VectorXd temp = matx2vecx(C);
            Eigen::VectorXd hParams = temp.head(temp.size()-2); //as last two are parameters for the GP not for Kernel. TODO: check if ell^2 or simply ells in MATLAB.
            Eigen::VectorXd policyParams;
            policyParams = push_back(policyParams,ps);//policyParams << ps , pt, hParams;
            policyParams = push_back(policyParams,pt);
            policyParams = push_back(policyParams,hParams);
            this->_params_from_file = policyParams;
        }

        void normalize(const Model& model)
        {

        }

        Eigen::VectorXd next(const Eigen::VectorXd state) const
        {
            Eigen::VectorXd policy_params;
            policy_params = _params;
            if (_random || policy_params.size() == 0) {
                 return Params::gp_policy::max_u() * (limbo::tools::random_vector(Params::action_dim()).array() * 2 - 1); //return random action
            }
            //---extract pseudo samples from parameters
            Eigen::VectorXd sample(sdim);
            std::vector<Eigen::VectorXd> pseudo_samples;
            for (int i=0; i<ps ; i++)
            {
                for(int j=0; j<sdim ;j++ )
                {
                        sample(j) = policy_params(i*sdim+j);
                }
                pseudo_samples.push_back(sample);
                //std::cout<<sample.transpose()<<std::endl;
            }
            //--- extract pseudo observations from parameres
            Eigen::VectorXd obs;
            std::vector<Eigen::VectorXd> pseudo_observations;
            obs = policy_params.segment(sdim*ps,ps);
            for (int i=0; i<obs.size(); i++)
            {
                Eigen::VectorXd temp = limbo::tools::make_vector(obs(i));
                pseudo_observations.push_back(temp);
                //std::cout<<temp.transpose()<<std::endl;
            }
            //--- extract hyperparameters from parameters
            Eigen::VectorXd ells(sdim);
            ells = policy_params.tail(sdim);
            //-- instantiating gp policy
            gp_t gp_policy_obj(5,1);
            //--- set hyperparameter ells in the kernel.
            gp_policy_obj.kernel_function().set_h_params(ells);
            //--- Compute the gp
            Eigen::VectorXd noises = Eigen::VectorXd::Constant(ps, Params::gp_policy::noise());
            gp_policy_obj.compute(pseudo_samples, pseudo_observations, noises); //TODO: Have to check the noises with actual PILCO
            //--- Query the GP with state
            Eigen::VectorXd action = gp_policy_obj.mu(state);
            action = action.unaryExpr([](double x) {return Params::gp_policy::max_u() * (9 * std::sin(x) / 8.0 + std::sin(3 * x) / 8.0);});
            return action;
        }
        void set_random_policy()
        {
            _random = true;
        }
        bool random() const
        {
            return _random;
        }
        void set_params(const Eigen::VectorXd& params)
        {
            _random = false;
            _params = params;
            if(Params::gp_policy::paramFromFile())
                _params = _params_from_file;
        }
        Eigen::VectorXd params() const
        {
            if (_random || _params.size() == 0)
                return limbo::tools::random_vector((sdim+1)*ps+sdim);//TODO: set the proper bounds here
            return _params;
        }
        Eigen::VectorXd _params;
        bool _random;

        Model* _model;
        Eigen::VectorXd _means;
        Eigen::MatrixXd _sigmas;
        Eigen::VectorXd _limits;
        Eigen::VectorXd _params_from_file;
    };
}
#endif
