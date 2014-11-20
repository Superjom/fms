#ifndef _FMS_KL_FM_H_
#define _FMS_KL_FM_H_
#include "./base_fm.h"

namespace fms {

template<typename DataType=IDData>
using KL_FM = BaseFM<KLdistSGD, AdaGradFMParam, DataType>;

}; // end namespace fms

#endif
