#ifndef _FMS_PAIR_FM_H_
#define _FMS_PAIR_FM_H_
#include "../data.h"
#include "../core/all.h"
#include "../utils/all.h"
#include "./kl_fm.h"

namespace fms {

template <typename DataType, typename InsType>
using PairFM = BaseFM<PairSGD, AdaGradFMParam, DataType, InsType>;

}; // end namespace fms


#endif
