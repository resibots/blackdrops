#ifndef MEDROPS_GP_POLICY_NEW_HPP
#define MEDROPS_GP_POLICY_NEW_HPP

#include <limbo/tools/random_generator.hpp>
#include <Eigen/Core>
#include <limbo/tools/macros.hpp>
#include <limbo/limbo.hpp>
#include <limbo/tools.hpp>

namespace medrops {



    template <typename Params>
    struct GPPolicyNew {

        using kernel_t = limbo::kernel::SquaredExpARD<Params>;
        using mean_t = limbo::mean::Data<Params>;
        size_t sdim = Params::nn_policy::state_dim(); //input dimension
        size_t ps = Params::gp_policy::pseudo_samples(); //total observations
        using gp_t = limbo::model::GP<Params, kernel_t, mean_t>;
        //gp_t gp_policy_obj;

        GPPolicyNew()
        {
            _random = false;
             size_t sdim = Params::nn_policy::state_dim();
             size_t ps = Params::gp_policy::pseudo_samples();
             _params = Eigen::VectorXd::Zero((sdim+1)*ps+sdim);

        }

        Eigen::VectorXd next(const Eigen::VectorXd state) const
        {
            if (_random || _params.size() == 0) {
                return Params::linear_policy::max_u() * (limbo::tools::random_vector(Params::action_dim()).array() * 2 - 1); //return max action
            }

            //---extract pseudo samples from parameters
            Eigen::VectorXd sample(sdim);
            std::vector<Eigen::VectorXd> pseudo_samples;
            for (int i=0; i<ps ; i++)
            {
                for(int j=0; j<sdim ;j++ )
                {
                    sample(j) = _params(i*sdim+j)*2*Params::gp_policy::max_sample()-Params::gp_policy::max_sample();
                }
                pseudo_samples.push_back(sample);
            }

            //--- extract pseudo observations from parameres
            Eigen::VectorXd obs;
            std::vector<Eigen::VectorXd> pseudo_observations;
            //std::cout<<"Hahaha _param size: "<<_params.size()<<std::endl;
            //std::cout<<"Start : "<<sdim*ps<<" Ends: "<<sdim*ps+ps-1<<std::endl;
            obs = _params.segment(sdim*ps,ps).array()*2*Params::gp_policy::max_u()-Params::gp_policy::max_u();  //NOTE: Code breaks here
            //std::cout<<"Hahaha ends: "<<obs.size()<<" "<<pseudo_samples.size()<<std::endl;

            for (int i=0; i<obs.size(); i++)
            {
                Eigen::VectorXd temp = limbo::tools::make_vector(obs(i));
                pseudo_observations.push_back(temp);
            }

            //std::cout << " Psedo obs size " << pseudo_observations.size() << std::endl;

            //--- extract hyperparameters from parameters
            Eigen::VectorXd ells(sdim);
            ells = _params.tail(sdim).array()*2*10-10; //Scaling between 10 and -10
            ells = ells.array().log();  //TODO: check if ells should be bounded.NOTE: [0,1] bounded parameters.


            //-- instantiating gp policy
            gp_t gp_policy_obj;

            //--- set hyperparameter ells in the kernel.
            gp_policy_obj.kernel_function().set_h_params(ells);

            //--- Compute the gp
            Eigen::VectorXd noises = Eigen::VectorXd::Constant(ps, Params::gp_policy::noise());
            gp_policy_obj.compute(pseudo_samples, pseudo_observations, noises); //TODO: Have to check the noises with actual PILCO

            //--- Query the GP with state
            std::tuple<Eigen::VectorXd, double> result;
            result = gp_policy_obj.query(state);
            auto action = std::get<0>(result); //extracticting only the mean
            action = action.unaryExpr([](double x) {return Params::gp_policy::max_u() * (9 * std::sin(x) / 8.0 + std::sin(3 * x) / 8.0);});
            return action;
            //return limbo::tools::make_vector(1);
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
            _params = params;
        }

        Eigen::VectorXd params() const
        {
            if (_random || _params.size() == 0)
                // return limbo::tools::random_vector((Params::linear_policy::state_dim() + 1) * Params::action_dim());
                return limbo::tools::random_vector(
                    (Params::linear_policy::state_dim()+2) *
                    Params::gp_policy::pseudo_samples() + 1);
            return _params;
        }

        Eigen::VectorXd _params;
        bool _random;
    };
}

#endif
