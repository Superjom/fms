#ifndef _CORE_SGD_BASESGD_H_
#define _CORE_SGD_BASESGD_H_
namespace fms {

template <class InstType=Instance>
class BaseSGD : public VirtualObject {
    virtual double learn_instance(const InstType &ins) = 0;
};

};  // end namespace fms

#endif
