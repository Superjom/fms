#ifndef _FMS_CORE_SGD_GRAD_VALUE_H_
#define _FMS_CORE_SGD_GRAD_VALUE_H_
#include "../utils/all.h"
#include "./common.h"
#include "./VirtualObject.h"
/*
 * 记录梯度
 */
namespace fms {

struct SGDGradValue {
private:
    double _lr_g = 0.0;
    Vec _fm_g;
    double _n = 0.0;

public:
    explicit SGDGradValue() { }
    explicit SGDGradValue(int dim) {
        init(dim);
    }
    void init(int dim) {
        _lr_g = 0.0;
    	_fm_g.init(dim);
    }
    explicit SGDGradValue(double lr_g, Vec fm_g) {
        _lr_g = lr_g;
        _fm_g = std::move(fm_g);
        _n  = 0.0;
    }
    void merge_with(const SGDGradValue &other) {
        //CHECK(!_fm_g.empty());
        //CHECK(!other.fm_g().empty());
        _lr_g += other.lr_g();
        _fm_g += other.fm_g();
        if (other.n() < 0.5) _n ++;
        else _n += other.n();
    }
    SGDGradValue normed() const {
        //CHECK(! _fm_g.empty());
        CHECK(_n > 0.0);
        SGDGradValue tmp(_fm_g.size());
        tmp._lr_g = _lr_g / _n;
        tmp._fm_g = _fm_g / _n;
        return std::move(tmp);
    }
    void reset() {
        _lr_g = 0.0;
        for(auto it = _fm_g.begin(); it != _fm_g.end(); ++it) {
            *it = 0.0;
        }
        _n = 0.0;
    }
    friend std::ostream & operator<< (std::ostream &os, const SGDGradValue &other) {
        os << "lr_g:\t" << other._lr_g << "\tn:\t" << other._n;
        for(index_t i = 0; i < other._fm_g.size(); ++ i) {
            os << "\t" << other._fm_g[i];
        }
        os << endl;
        return os;
    }
    double lr_g() const {
        return _lr_g;
    }
    void set(double lr_g, const Vec &fm_g) {
        _lr_g = lr_g;
        _fm_g = fm_g;
    }
    const Vec &fm_g() const {
        return _fm_g;
    }
    Vec &fm_g() {
        return _fm_g;
    }
    double n() const {
        return _n;
    }
};

}; // end namespace
#endif
