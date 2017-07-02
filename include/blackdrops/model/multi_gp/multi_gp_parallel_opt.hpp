#ifndef BLACKDROPS_MODEL_MULTI_GP_MULTI_GP_PARALLEL_OPT_HPP
#define BLACKDROPS_MODEL_MULTI_GP_MULTI_GP_PARALLEL_OPT_HPP

#include <Eigen/binary_matrix.hpp>
#include <limbo/model/gp/hp_opt.hpp>
#include <limbo/model/gp/kernel_lf_opt.hpp>

namespace blackdrops {
    namespace model {
        namespace multi_gp {
            template <typename Params, typename Optimizer = limbo::opt::Rprop<Params>>
            struct MultiGPParallelLFOpt : public limbo::model::gp::HPOpt<Params, Optimizer> {
            public:
                template <typename GP>
                void operator()(GP& gp)
                {
                    this->_called = true;
                    auto& gps = gp.gp_models();
                    // for (auto& small_gp : gps)
                    tbb::parallel_for(size_t(0), gps.size(), size_t(1), [&](size_t i) {
                        Optimizer hp_optimize;
                        hp_optimize(gps[i]);
                    });
                }
            };
        }
    }
}
#endif
