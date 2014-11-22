#ifndef _CORE_SGD_PAIRSGD_H_
#define _CORE_SGD_PAIRSGD_H_
#include <map>
#include "../common.h"
#include "BaseSGD.h"

namespace fms {
/*
 * pairwise Rank 的方式训练
 * 对order的学习
 */
class PairSGD : public BaseSGD<ListInstance> {
public:
    enum comp_t {    // pair两个元素间的大小关系
        GREATER = 2,
        EQUAL = 1,
        LOWER = 0
    };
    explicit PairSGD() { }
    explicit PairSGD(AdaGradFMParam &fm) : _fm(&fm)
    {
        CHECK(this->_fm != nullptr);
    }
    void set_fm(AdaGradFMParam &fm) { _fm = &fm; }
    double learn_instance(const ListInstance &list_ins) {
        //cout <<"learn instance:\t" << list_ins << endl;
        Cost total_cost;
        for (index_t i = 0; i < list_ins.list.size()-1; ++i) {
            for(index_t j = i+1; j < list_ins.list.size(); ++j) {
                Vec pre_x_v_sum(_fm->dim());
                Vec next_x_v_sum(_fm->dim());

                Instance pre(list_ins.list[i]);
                Instance next(list_ins.list[j]);
                merge_instance(list_ins.prefix, pre);
                merge_instance(list_ins.prefix, next);
                // 添加共同prefix
                /*
                for(auto it = list_ins.prefix.begin(); it != list_ins.prefix.end(); ++it) {
                    pre.feas.push_back(*it);
                    next.feas.push_back(*it);
                }
                */
                comp_t label = GREATER;
                if (pre.target < next.target) {
                    label = LOWER;
                } else if (pre.target == next.target) {
                    label = EQUAL;
                }
                //cout << "label:\t" << label << endl;
                double q, o12;
                double cost = forward(label, pre, next, pre_x_v_sum, next_x_v_sum, q, o12);
                //cout << "cost\t" << cost << endl;
                if(std::isnan(cost)) {
                    // 忽略 Nan 记录
                    continue;
                }
                backward(label, pre, next, pre_x_v_sum, next_x_v_sum, q, o12);
                total_cost.cumulate(cost);
            }
        }
        return total_cost.norm();
    }
    double forward(const ListInstance &list_ins) {
        Cost total_cost;
        for (index_t i = 0; i < list_ins.list.size()-1; ++i) {
            for(index_t j = i+1; j < list_ins.list.size(); ++i) {
                Vec pre_x_v_sum(_fm->dim());
                Vec next_x_v_sum(_fm->dim());

                Instance pre(list_ins.list[i]);
                Instance next(list_ins.list[j]);
                // 添加共同prefix
                merge_instance(list_ins.prefix, pre);
                merge_instance(list_ins.prefix, next);
                //CHECK(list_ins.list[i].feas.size() + list_ins.prefix.size() == pre.feas.size());
                comp_t label = GREATER;
                if (pre.target < next.target) {
                    label = LOWER;
                } else if (pre.target == next.target) {
                    label = EQUAL;
                }
                double q, o12;
                double cost = forward(label, pre, next, pre_x_v_sum, next_x_v_sum, q, o12);
                if(std::isnan(cost)) {
                    // 忽略 Nan 记录
                    return cost;
                }
                //backward(label, pre, next, pre_x_v_sum, next_x_v_sum, q, o12);
                total_cost.cumulate(cost);
            }
        }
        return total_cost.norm();
    }
protected:
    double forward(comp_t label, const Instance &pre, const Instance &next, Vec &pre_x_v_sum, Vec &next_x_v_sum, double &q, double &o12) {
        double o1 = cal_ins_fm_score(pre, pre_x_v_sum);
        double o2 = cal_ins_fm_score(next, next_x_v_sum);
        //cout << "o1:o2\t" << o1 << "\t" << o2 << endl;
        o12 = o1 - o2;
        //cout << "o12\t" << o12 << endl;
        q = exp(o12) / (exp(o12) + 1.0);
        //const double &p = (double)label / 2.0;
        /*
         * 根据pair的对错进行预测
         * TODO 具体的 EQUAL等标准需要一个软性的划分
         */
        //CHECK(!std::isnan(o12));
        if(std::isnan(o12)) return o12;
        double cost = 1.0;
        if( (label == GREATER   && o12 >  0.0) || \
            (label == EQUAL     && o12 == 0.0) || \
            (label == LOWER     && o12 <  0.0) ) cost = 0.0;
        return cost;
    }
    void backward(comp_t label, const Instance &pre, const Instance &next, Vec &pre_x_v_sum, Vec &next_x_v_sum, double q, double o12) {
        const double &p = (double)label / 2.0;
        double exp_o12 = exp(o12);
        double grad_C_q = - p / q + (1 - p) / (1 - q);
        double grad_q_o12 = exp_o12 / ( exp_o12 * exp_o12 + 2 * exp_o12 + 1);
        double grad_o12_o1 = 1.0, grad_o12_o2 = -1.0;
        double grad_C_o1 = grad_C_q * grad_q_o12 * grad_o12_o1;
        double grad_C_o2 = grad_C_q * grad_q_o12 * grad_o12_o2;

        // pre
        backward_ins(grad_C_o1, pre, pre_x_v_sum);
        backward_ins(grad_C_o2, next, next_x_v_sum);
    }

protected:
    double cal_ins_fm_score(const Instance &ins, Vec &x_v_sum) {
        double x_v2sum{0.0}; // (xv)^2
        double lr_score{0.0};
        //const double &p = ins.target;
        for(auto it = ins.feas.begin(); it != ins.feas.end(); ++it) {
            const index_t &key = it->key;
            FMValue &feature = _fm->feature(key);
            const double &x = it->value;
            const double &w = feature.lr_w;
            Vec &v = feature.fm_w;
            //cout << "w\t" << w << endl;

            lr_score += w * x;
            x_v_sum += (x * v);
            x_v2sum += dot((x * v), (x * v));
        }
        double fm_score = 0.5 * (dot(x_v_sum, x_v_sum) - x_v2sum);
        //cout << "fm_score\t" << fm_score << endl;
        //cout << "lr_score\t" << lr_score << endl;
        //lr_score = 0.0;
        double o = lr_score + fm_score;
        //cout << "lr_score\t" << lr_score << endl;
        //cout << "o\t" << o << endl;
        // TODO merge the lr_score !!!
        //double o = fm_score;
        return o;
    }
    void backward_ins(double grad_C_o, const Instance &ins, Vec &x_v_sum) {
        SGDGradValue grad(_fm->dim());
        for(auto it = ins.feas.begin(); it != ins.feas.end(); ++it) {
            const index_t &key = it->key;
            const double &x = it->value;
            double grad_o_lr = x;
            FMValue &feature = _fm->feature(key);
            Vec &v = feature.fm_w;
            Vec grad_o_v(_fm->dim());
            grad_o_v = x * (x_v_sum - x * v);
            double grad_C_lr = grad_C_o * grad_o_lr;
            //cout << "grad_C_lr\t" << grad_C_lr << endl;
            Vec grad_C_v = grad_C_o * grad_o_v;
            //cout << "grad_C_v\t" << grad_C_v << endl;
            grad.set(grad_C_lr, grad_C_v);
            _fm->batch_commit(key, grad);
            grad.reset();
        } // end for 
    }
private:
    AdaGradFMParam *_fm = nullptr;
};  // end class 



}; // end namespace fms

#endif
