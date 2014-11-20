#ifndef _FMS_PAIR_SIGMOID_FM_H_
#define _FMS_PAIR_SIGMOID_FM_H_
#include "../data.h"
#include "../core/all.h"
#include "../utils/all.h"
#include "./base_fm.h"

namespace fms {

template <typename DataType, typename InsType>
using PairSigmoidFM = BaseFM<PairSigmoidSGD, AdaGradFMParam, DataType, InsType>;

}; // end namespace fms


#endif

