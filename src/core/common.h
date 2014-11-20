#ifndef _FMS_CORE_COMMON_H_
#define _FMS_CORE_COMMON_H_
#include <map>
#include "../utils/all.h"
namespace fms {

// 适合回归任务
// 数据单个记录
struct Instance {
    struct Item {
        index_t key;
        double value;
        Item(index_t key, double value) : key(key), value(value) { }
        friend std::ostream &operator<< (std::ostream &os, const Item &other) {
            os << other.key << ":" << other.value;
            return os;
        }
    };
    Instance () { }
    Instance(const Instance &other) {
        target = other.target;
        feas = other.feas;
    }
    double target{0};
    std::vector<Item> feas;     // feature
    friend std::ostream &operator<< (std::ostream &os, const Instance &other) {
        os << "target:" << other.target << "feature:";
        for (auto it = other.feas.begin(); it != other.feas.end(); ++it) {
            os << *it << " ";
        }
        os << std::endl;
        return os;
    }
};

// 适合Rank任务 
// list 方式存储的order 利用prefix实现数据压缩
/* 
 * 输入是list
 * 第一个field是共同的前缀 id:v id:v id:v , 如果没用共同的head，可以设置为空
 * 后面跟上多个 Instance
 * 每个Instance数据格式如下： target id:v id:v id:v
 * 不同的field之间用\t分割
 * 12:1.0 2:3.1 \t 0.7 13:1.0 \t 0.9 1:0.5 2:0.3
 */
struct ListInstance {   
    std::vector<Instance::Item> prefix;
    std::vector< Instance > list;

    friend std::ostream& operator<< (std::ostream &os, const ListInstance &other) {
        os << "prefix:";
        for (auto it = other.prefix.begin(); it != other.prefix.end(); ++it) {
            os << *it << " ";
        }
        os << "instances:\t";
        for (auto it = other.list.begin(); it != other.list.end(); ++it) {
            os << *it << "\t";
        }
        os << endl;
        return os;
    }
};




}; // namespace fms
#endif
