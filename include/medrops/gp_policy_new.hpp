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
        //using mean_t = limbo::mean::Data<Params>;
        using mean_t = limbo::mean::Constant<Params>;
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
                return Params::linear_policy::max_u() * (limbo::tools::random_vector(Params::action_dim()).array() * 2 - 1); //return random action
            }

            //---extract pseudo samples from parameters
            Eigen::VectorXd sample(sdim);
            std::vector<Eigen::VectorXd> pseudo_samples;
            //std::cout<<"_params: "<<_params<<std::endl;
            for (int i=0; i<ps ; i++)
            {
                for(int j=0; j<sdim ;j++ )
                {
                    if(j==3 || j==4) //as cos and sine bounded between [-1,1]. No need to scale.
                        sample(j) = _params(i*sdim+j);
                    else
                        sample(j) = _params(i*sdim+j)*2*Params::gp_policy::max_sample()-Params::gp_policy::max_sample(); //scaling bounded params otherwise
                    //std::cout<<"Sample("<<j<<") : "<<sample(j)<<std::endl;
                    //std::cout<<"Sample orig: "<<_params(i*sdim+j)<<std::endl;
                }
                //std::getchar();
                pseudo_samples.push_back(sample);
            }

            //--- extract pseudo observations from parameres
            Eigen::VectorXd obs;
            std::vector<Eigen::VectorXd> pseudo_observations;
            obs = _params.segment(sdim*ps,ps).array()*2*Params::gp_policy::max_u()-Params::gp_policy::max_u();
            //std::cout<<"Pseudo Obs: "<<obs<<std::endl;
            //std::getchar();
            for (int i=0; i<obs.size(); i++)
            {
                Eigen::VectorXd temp = limbo::tools::make_vector(obs(i));
                pseudo_observations.push_back(temp);
            }


            //--- extract hyperparameters from parameters
            Eigen::VectorXd ells(sdim);
            ells = _params.tail(sdim).array()*10; //Scaling between 0 to 10
            ells = ells.array().log();  //TODO: check if ells should be bounded.NOTE: [0,1] bounded parameters.
            //std::cout<<"Ells : "<<ells<<std::endl;
            //std::getchar();

            //-- instantiating gp policy
            gp_t gp_policy_obj;

            //--- set hyperparameter ells in the kernel.
            gp_policy_obj.kernel_function().set_h_params(ells);

            //--- Compute the gp
            Eigen::VectorXd noises = Eigen::VectorXd::Constant(ps, Params::gp_policy::noise());
            gp_policy_obj.compute(pseudo_samples, pseudo_observations, noises); //TODO: Have to check the noises with actual PILCO

            //--- Query the GP with state
            std::tuple<Eigen::VectorXd, double> result;
            //std::cout<<"State : "<<state<<std::endl;

            // result = gp_policy_obj.query(state);
            Eigen::VectorXd action = gp_policy_obj.mu(state);//std::get<0>(result); //extracticting only the mean
            //std::cout<<"Action : "<<action<<std::endl;
            action = action.unaryExpr([](double x) {return Params::gp_policy::max_u() * (9 * std::sin(x) / 8.0 + std::sin(3 * x) / 8.0);});
            //std::cout<<"Action processed : "<<action<<std::endl;
            //std::getchar();
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
            //std::cout<<"called "<<params.size()<<std::endl;
            //std::getchar();
        }

        Eigen::VectorXd params() const
        {
            if (_random || _params.size() == 0)
                return limbo::tools::random_vector((sdim+1)*ps+sdim);
            return _params;
        }

        Eigen::VectorXd _params;
        bool _random;
    };
}

#endif
