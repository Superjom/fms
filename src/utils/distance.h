#ifndef _UTILS_DISTANCE_H_
#define _UTILS_DISTANCE_H_
#include <cmath>

namespace fms {

inline double KLdistance(double p, double q) {
    return p * std::log( p / q) + q * std::log( q / p);
}

inline double RMSE(double p, double q) {
    return std::pow((p - q), 2);
}

};  // end namespace fms

#endif
