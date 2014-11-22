#ifndef _FMS_CORE_ADA_FMC_PARAM_H_
#define _FMS_CORE_ADA_FMC_PARAM_H_
#include <cmath>
#include "../utils/all.h"
#include "./common.h"
#include "./VirtualObject.h"
#include "./FMCParam.h"
#include "./SGDGradValue.h"

namespace fms {
/* min-batch-AdaSGD方式更新参数
 * 综合 Param 和 Grad 参数
 */
class AdaGradFMParam : public FMParam {
public:
    explicit AdaGradFMParam() { }
    explicit AdaGradFMParam(int dim, index_t num_feas, double init_learning_rate=0.1) : FMParam(dim, num_feas), 
        _init_learning_rate(init_learning_rate)
    { 
        init(dim, num_feas);
    }
    ~AdaGradFMParam() {
        batch_push();
    }
    void init(int dim, index_t num_feas, double init_learning_rate=0.1) {
        _init_learning_rate = init_learning_rate;
        _dim = dim;
        _num_feas = num_feas;
        for( index_t i = 0; i < num_feas; ++i ) {
            SGDGradValue tmp(dim);
            _sgd_grad.push_back(std::move(tmp));
        }
    }
    // 单次的sgd
    void batch_commit(index_t key, const SGDGradValue &grad) {
        CHECK(key < _sgd_grad.size());
        _sgd_grad[key].merge_with(grad);
    }
    // batch 方式更新参数
    void batch_push() {
        for(index_t key = 0; key < _sgd_grad.size(); ++key) {
            if(_sgd_grad[key].n() < 0.5) continue;
            SGDGradValue normed_grad = std::move(_sgd_grad[key].normed());
            FMValue &fea = feature(key);
            // update lr weight
            if(normed_grad.lr_g() != 0.0) { // fix the lr_g NaN bug
                fea.lr2sum += pow(normed_grad.lr_g(), 2);
                fea.lr_w -= _init_learning_rate * normed_grad.lr_g() / std::sqrt(fea.lr2sum);
            }
            // TODO remove this check
            //cout << "key\t" << key << "\tlr_w\t" << fea.lr_w << "\tlr2sum\t" << fea.lr2sum << endl;
            // update fm weight
            fea.fm2sum += (normed_grad.fm_g() * normed_grad.fm_g());
            fea.fm_w -= _init_learning_rate * normed_grad.fm_g() / sqrt(fea.fm2sum);
        }
        batch_reset();
    }
    void batch_reset() {
        for(auto it = _sgd_grad.begin(); it != _sgd_grad.end(); ++it) {
            it->reset();
        }
    }
private:
    std::vector<SGDGradValue> _sgd_grad;
    double _init_learning_rate = 0.1;
}; // class AdaGradFMParam


}; // end namespace 

#endif
