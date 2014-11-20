#ifndef _UTILS_COST_H_
#define _UTILS_COST_H_
#include "../core/VirtualObject.h"

namespace fms {

// 统计全局的Cost
class Cost : public VirtualObject {
public:
    void cumulate(double cost) {
        cumulate_cost += cost;
        n++;
    }
    double norm() {
        return cumulate_cost / n;
    }
    void reset() {
        cumulate_cost = 0;
        n = 0;
    }


private:
    double cumulate_cost = 0.0;
    double n = 0;
};

}; // end namespace fms

#endif
