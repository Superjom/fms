#ifndef CORE_VIRTUAL_OBJECT_H_
#define CORE_VIRTUAL_OBJECT_H_

namespace fms {

class Object {
};

class NoncopyableObject : public Object {
public:
    NoncopyableObject() = default;
    NoncopyableObject(const NoncopyableObject& ) = delete;
    NoncopyableObject& operator= (const NoncopyableObject& ) = delete;
};

class VirtualObject : public NoncopyableObject {
public:
    virtual ~VirtualObject() = default;
};

}; // end namespace fms

#endif
