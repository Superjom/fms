#ifndef _CORE_SGD_KLDISTSGD_H_
#define _CORE_SGD_KLDISTSGD_H_
#include "BaseSGD.h"
namespace fms {

// 采用KL距离的方式训练
class KLdistSGD : public BaseSGD<Instance> {
public:
    explicit KLdistSGD() { }
    explicit KLdistSGD(AdaGradFMParam &fm) : fm(&fm)
    {
        CHECK(this->fm != nullptr);
    }
    void set_fm(AdaGradFMParam &fm) {
        this->fm = &fm;
    }
    double learn_instance(const Instance &ins) {
        Vec x_v_sum(fm->dim());
        double q, y;
        double cost = forward(ins, x_v_sum, q, y);
        if(std::isnan(cost)) {   // 如果溢出 则跳过该记录 不进行backward
            return cost;
        }
        backward(ins, x_v_sum, q, y);
        return cost;
    }
    double forward(const Instance &ins) {
        Vec x_v_sum(fm->dim());
        double q, y;
        double cost = forward(ins, x_v_sum, q, y);
        if(std::isnan(cost)) {
            // 如果溢出 那么取平均数
            q = 0.5;
        }
        cost = KLdistance(ins.target, q);
        return cost;
    }
    double forward(const Instance &ins, Vec &x_v_sum, double &q, double &y) {
        double x_v2sum{0.0}; // (xv)^2
        double lr_score = 0.0;
        const double &p = ins.target;
        for(auto it = ins.feas.begin(); it != ins.feas.end(); ++it) {
        //for(index_t i = 0; i < ins.feas.size(); ++i) {
            const index_t &key = it->key;
            FMValue &feature = fm->feature(key);
            //cout << "feature\t" << feature << endl;
            const double &x = it->value;
            const double &w = feature.lr_w;
            //cout << "w\t" << w << endl;
            Vec &v = feature.fm_w;

            lr_score += w * x;
            x_v_sum += (x * v);
            x_v2sum += dot((x * v), (x * v));
        }
        double fm_score = 0.5 * (dot(x_v_sum, x_v_sum) - x_v2sum);
        y = lr_score + fm_score;
        q = 1 / (1 + exp(-y));
        return KLdistance(p, q);
    }
    void backward(const Instance &ins, Vec &x_v_sum, double q, double y) {
        const double &p = ins.target;
        double grad_KL_q = - p / q + log(q / p) + 1;
        //cout << "grad_KL_q\t" << grad_KL_q << endl;
        double grad_q_y  = exp(y) / pow( (1 + exp(y)), 2);
        //cout << "grad_q_y\t" << grad_q_y << endl;
        SGDGradValue grad(fm->dim());
        //double *fm_g = &grad.fm_g()[0];
        for(auto it = ins.feas.begin(); it != ins.feas.end(); ++it) {
            const index_t &key = it->key;
            const double &x = it->value;
            double grad_y_lr = x;
            FMValue &feature = fm->feature(key);
            Vec &v = feature.fm_w;
            Vec grad_y_v(fm->dim());
            grad_y_v = x * (x_v_sum - x * v);
            double grad_KL_lr = grad_KL_q * grad_q_y * grad_y_lr;
            Vec grad_KL_v = grad_KL_q * grad_q_y * grad_y_v;
            //cout << "grad_KL_lr\t" << grad_KL_lr << endl;
            //cout << "grad_KL_v\t" << grad_KL_v << endl;
            grad.set(grad_KL_lr, grad_KL_v);
            //cout << "grad\t" << grad << endl;
            fm->batch_commit(key, grad);
            grad.reset();
        } // end for 
    }
private:
    AdaGradFMParam *fm = nullptr;
};  // class KLdistSGD



};   // end namespace fms

#endif
